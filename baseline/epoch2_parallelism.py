import argparse
import logging
import math
import os
import random
import threading
import time

import GPUtil
import numpy as np
import torch
import torch.distributed
import torch.nn
import torch.nn.parallel
import torch.utils.data
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import BatchSampler, SequentialSampler
from tqdm import tqdm

import gnnflow.cache as caches
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from config import get_default_config
import torch.distributed as dist
from gnnflow.data import (EdgePredictionDataset,
                          RandomStartBatchSampler, default_collate_ndarray)
from modules.sampler import DistributedBatchSampler
from gnnflow.models.dgnn import DGNN
from gnnflow.models.gat import GAT
from gnnflow.models.graphsage import SAGE
from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import (DstRandEdgeSampler, EarlyStopMonitor,
                           build_dynamic_graph, get_pinned_buffers,
                           get_project_root_dir, load_dataset,
                           mfgs_to_cuda)
from modules.util import save_memory, put_memory, load_feat

datasets = ['REDDIT', 'GDELT', 'LASTFM', 'MAG', 'MOOC', 'WIKI']
model_names = ['TGN', 'TGAT', 'DySAT', 'GRAPHSAGE', 'GAT']
cache_names = sorted(name for name in caches.__dict__
                     if not name.startswith("__")
                     and callable(caches.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=model_names, default='TGN',
                    help="model architecture" + '|'.join(model_names))
parser.add_argument("--data", choices=datasets, required=True,
                    help="dataset:" + '|'.join(datasets))
parser.add_argument("--epoch", help="maximum training epoch",
                    type=int, default=5)
parser.add_argument("--lr", help='learning rate', type=float, default=0.0001)
parser.add_argument("--num-workers", help="num workers for dataloaders",
                    type=int, default=1)
parser.add_argument("--num-chunks", help="number of chunks for batch sampler",
                    type=int, default=1)
parser.add_argument("--print-freq", help="print frequency",
                    type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ingestion-batch-size", type=int, default=10000000,
                    help="ingestion batch size")

# optimization
parser.add_argument("--cache", choices=cache_names, default='LRUCache', help="feature cache:" +
                    '|'.join(cache_names))
parser.add_argument("--edge-cache-ratio", type=float, default=0,
                    help="cache ratio for edge feature cache")
parser.add_argument("--node-cache-ratio", type=float, default=0,
                    help="cache ratio for node feature cache")
parser.add_argument("--snapshot-time-window", type=float, default=0,
                    help="time window for sampling")

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
logging.info(args)

checkpoint_path = os.path.join(get_project_root_dir(),
                               '{}.pt'.format(args.model))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

training = True


def gpu_load():
    global training
    # time.sleep(5)
    # while True:
    #     # stop when training is done
    #     # use a global variable to stop the thread
    #     if not training:
    #         break
    #     gpus = GPUtil.getGPUs()
    #     avg_load = sum([gpu.load for gpu in gpus]) / len(gpus)
    #     logging.info("GPU load: {:.2f}%".format(avg_load * 100))
    #     time.sleep(1)


def evaluate(dataloader, sampler, model, criterion, cache, device, local_group):
    dist.barrier(local_group)
    if args.local_rank == 0:
        input = save_memory(model)
    model.eval()
    model.reset()
    val_losses = list()
    aps = list()
    aucs_mrrs = list()
    cnt = 0
    max_iteration = 300
    # start_id = random.randint(0, int(len(dataloader)//args.local_world_size-max_iteration-5))
    start_id = 0
    with torch.no_grad():
        total_loss = 0
        for target_nodes, ts, eid in dataloader:
            if start_id > 0:
                start_id -= 1
                continue
            cnt += 1
            if cnt>max_iteration:
                break
            mfgs = sampler.sample(target_nodes, ts)
            mfgs_to_cuda(mfgs, device)
            mfgs = cache.fetch_feature(
                mfgs, eid)
            if args.use_memory:
                b = mfgs[0][0]
                model.memory.prepare_input(b)
                model.last_updated = model.memory_updater(b)

            pred_pos, pred_neg = model(mfgs)

            if args.use_memory:
                # NB: no need to do backward here
                # use one function
                model.memory.update_mem_mail(
                    **model.last_updated, edge_feats=cache.target_edge_features,
                    neg_sample_ratio=1)

            total_loss += criterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)),
                torch.zeros(pred_neg.size(0))], dim=0)
            aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            aps.append(average_precision_score(y_true, y_pred))

        val_losses.append(float(total_loss))
    
    ap = float(torch.tensor(aps).mean())
    auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    dist.barrier(local_group)
    if args.local_rank == 0:
        put_memory(input, model)
    return ap, auc_mrr


def main():
    args.distributed = int(os.environ.get('WORLD_SIZE', 0)) > 1
    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group('nccl')
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        args.node_rank = args.rank//args.local_world_size
        args.num_nodes = args.world_size // args.local_world_size
        for i in range(args.num_nodes):
            if i==args.node_rank:
                local_group = dist.new_group([rank for rank in range(args.local_world_size*args.node_rank, args.local_world_size*(args.node_rank+1))])
            else:
                dist.new_group([rank for rank in range(args.local_world_size*i, args.local_world_size*(i+1))])
    else:
        args.local_rank = args.rank = 0
        args.local_world_size = args.world_size = 1
    num_gpus = torch.tensor(data=(1), device=f'cuda:{args.local_rank}')
    dist.all_reduce(num_gpus, op=dist.ReduceOp.SUM)

    logging.info("rank: {}, world_size: {}".format(args.rank, args.world_size))

    model_config, data_config = get_default_config(args.model, args.data)
    if model_config["snapshot_time_window"] > 0 and args.data == "GDELT":
        model_config["snapshot_time_window"] = 25
    else:
        model_config["snapshot_time_window"] = args.snapshot_time_window
    logging.info("snapshot_time_window's value is {}".format(model_config["snapshot_time_window"]))
    args.use_memory = model_config['use_memory']

    if args.distributed:
        # graph is stored in shared memory
        data_config["mem_resource_type"] = "shared"

    train_data, val_data, test_data, full_data = load_dataset(args.data, data_dir='/data/TGL')
    train_rand_sampler = DstRandEdgeSampler(
        train_data['dst'].to_numpy(dtype=np.int32))
    val_rand_sampler = DstRandEdgeSampler(
        full_data['dst'].to_numpy(dtype=np.int32))
    test_rand_sampler = DstRandEdgeSampler(
        full_data['dst'].to_numpy(dtype=np.int32))

    train_ds = EdgePredictionDataset(train_data, train_rand_sampler)
    val_ds = EdgePredictionDataset(val_data, val_rand_sampler)
    test_ds = EdgePredictionDataset(test_data, test_rand_sampler)

    batch_size = model_config['batch_size']
    # NB: learning rate is scaled by the number of workers
    args.lr = args.lr * math.sqrt(args.world_size)
    logging.info("batch size: {}, lr: {}".format(batch_size, args.lr))

    if args.distributed:
        train_sampler = DistributedBatchSampler(
            SequentialSampler(train_ds), batch_size=batch_size,
            drop_last=False, rank=args.local_rank, world_size=args.local_world_size,
            num_chunks=args.num_chunks)
        val_sampler = DistributedBatchSampler(
            SequentialSampler(val_ds),
            batch_size=batch_size, drop_last=False, rank=args.local_rank,
            world_size=args.local_world_size)
        test_sampler = DistributedBatchSampler(
            SequentialSampler(test_ds),
            batch_size=batch_size, drop_last=False, rank=args.local_rank,
            world_size=args.local_world_size)
    else:
        train_sampler = RandomStartBatchSampler(
            SequentialSampler(train_ds), batch_size=batch_size, drop_last=False)
        val_sampler = BatchSampler(
            SequentialSampler(val_ds), batch_size=batch_size, drop_last=False)
        test_sampler = BatchSampler(
            SequentialSampler(test_ds),
            batch_size=batch_size, drop_last=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, sampler=train_sampler,
        collate_fn=default_collate_ndarray, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_ds, sampler=val_sampler,
        collate_fn=default_collate_ndarray, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_ds, sampler=test_sampler,
        collate_fn=default_collate_ndarray, num_workers=args.num_workers)

    dgraph = build_dynamic_graph(
        **data_config, device=args.local_rank)

    if args.distributed:
        torch.distributed.barrier()
    # insert in batch
    for i in tqdm(range(0, len(full_data), args.ingestion_batch_size)):
        batch = full_data[i:i + args.ingestion_batch_size]
        src_nodes = batch["src"].values.astype(np.int64)
        dst_nodes = batch["dst"].values.astype(np.int64)
        timestamps = batch["time"].values.astype(np.float32)
        eids = batch["eid"].values.astype(np.int64)
        dgraph.add_edges(src_nodes, dst_nodes, timestamps,
                         eids, add_reverse=False)
        if args.distributed:
            torch.distributed.barrier()

    num_nodes = dgraph.max_vertex_id() + 1
    num_edges = dgraph.num_edges()
    # put the features in shared memory when using distributed training
    node_feats, edge_feats = load_feat(
        args.data, data_dir='/data/TGL', shared_memory=args.distributed, rank = args.rank,
        local_rank=args.local_rank, local_world_size=args.local_world_size, local_group=local_group)

    dim_node = 0 if node_feats is None else node_feats.shape[1]
    dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

    device = torch.device('cuda:{}'.format(args.local_rank))
    logging.debug("device: {}".format(device))

    if args.model == "GRAPHSAGE":
        model = SAGE(dim_node, model_config['dim_embed'])
    elif args.model == 'GAT':
        model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=args.distributed)
    else:
        model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=args.distributed)
    model.to(device)

    sampler = TemporalSampler(dgraph, **model_config)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args.local_rank], find_unused_parameters=True)

    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(
        model_config['fanouts'], model_config['num_snapshots'], batch_size,
        dim_node, dim_edge)

    # Cache
    cache = caches.__dict__[args.cache](args.edge_cache_ratio, args.node_cache_ratio,
                                        num_nodes, num_edges, device,
                                        node_feats, edge_feats,
                                        dim_node, dim_edge,
                                        pinned_nfeat_buffs,
                                        pinned_efeat_buffs,
                                        None,
                                        False)

    # only gnnlab static need to pass param
    if args.cache == 'GNNLabStaticCache':
        cache.init_cache(sampler=sampler, train_df=train_data,
                         pre_sampling_rounds=2)
    else:
        cache.init_cache()

    logging.info("cache mem size: {:.2f} MB".format(
        cache.get_mem_size() / 1000 / 1000))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_e = train(train_loader, val_loader, sampler,
                   model, optimizer, criterion, cache, device, local_group)

    logging.info('Loading model at epoch {}...'.format(best_e))

    # ap, auc = evaluate(test_loader, sampler, model,
    #                    criterion, cache, device)
    # if args.distributed:
    #     metrics = torch.tensor([ap, auc], device=device)
    #     torch.distributed.all_reduce(metrics)
    #     metrics /= args.local_world_size
    #     ap, auc = metrics.tolist()
    #     if args.rank == 0:
    #         logging.info('Test ap:{:4f}  test auc:{:4f}'.format(ap, auc))

    if args.distributed:
        torch.distributed.barrier()


def train(train_loader, val_loader, sampler, model, optimizer, criterion,
          cache, device, local_group):
    global training
    best_ap = 0
    best_e = 0
    epoch_time_sum = 0
    early_stopper = EarlyStopMonitor()

    next_data = None

    def sampling(target_nodes, ts, eid):
        nonlocal next_data
        mfgs = sampler.sample(target_nodes, ts)
        next_data = (mfgs, eid)

    if args.local_rank == 0:
        gpu_load_thread = threading.Thread(target=gpu_load)
        gpu_load_thread.start()

    logging.info('Start training...')
    if args.distributed:
        torch.distributed.barrier()

    auc_list, tb_list, loss_list = [], [], []
    
    length = len(train_loader)
    per_length = int(length // args.local_world_size)
    print(f'debug, {int(per_length//args.num_nodes*args.node_rank)}')
    for _ in range(int(per_length//args.num_nodes*args.node_rank)):
        optimizer.zero_grad()
        syn_model(model, 0)
        optimizer.step()
    for e in range(args.epoch):
        start_time = time.time()
        model.train()
        cache.reset()
        if e > 0:
            model.reset()
        total_loss = 0
        cache_edge_ratio_sum = 0
        cache_node_ratio_sum = 0
        total_sampling_time = 0
        total_feature_fetch_time = 0
        total_memory_fetch_time = 0
        total_memory_update_time = 0
        total_memory_write_back_time = 0
        total_model_train_time = 0
        total_samples = 0
        epoch_time = 0

        train_iter = iter(train_loader)
        t1 = time.time()
        target_nodes, ts, eid = next(train_iter)
        if args.local_rank == 0:
            print(f'data load time = {(time.time()-t1):.2f}')
            
        mfgs = sampler.sample(target_nodes, ts)
        next_data = (mfgs, eid)

        sampling_thread = None

        iteration_now = args.local_rank

        i = 0
        epoch_time += time.time()-start_time
        while True:
            start_time = time.time()
            sampling_start_time = time.time()
            if sampling_thread is not None:
                sampling_thread.join()

            mfgs, eid = next_data
            num_target_nodes = len(eid) * 3

            # Sampling for next batch
            try:
                next_target_nodes, next_ts, next_eid = next(train_iter)
            except StopIteration:
                break
            if(iteration_now-args.local_rank+args.local_world_size + args.local_world_size >= len(train_loader)):
                break
            iteration_now += args.local_world_size
            # sampling(next_target_nodes, next_ts, next_eid)
            sampling_thread = threading.Thread(target=sampling, args=(
                next_target_nodes, next_ts, next_eid))
            sampling_thread.start()
            mfgs_to_cuda(mfgs, device)
            # Feature
            total_sampling_time += time.time()-sampling_start_time
            feature_start_time = time.time()
            mfgs = cache.fetch_feature(
                mfgs, eid)
            total_feature_fetch_time += time.time() - feature_start_time

            if args.use_memory:
                b = mfgs[0][0]
                memory_fetch_start_time = time.time()
                model.memory.prepare_input(b)
                total_memory_fetch_time += time.time() - memory_fetch_start_time

                memory_update_start_time = time.time()
                model.last_updated = model.memory_updater(b)
                total_memory_update_time += time.time() - memory_update_start_time

            # Train
            model_train_start_time = time.time()
            optimizer.zero_grad()
            pred_pos, pred_neg = model(mfgs)
            total_model_train_time += time.time() - model_train_start_time

            if args.use_memory:
                # NB: no need to do backward here
                with torch.no_grad():
                    # use one function
                    memory_write_back_start_time = time.time()
                    model.memory.update_mem_mail(
                        **model.last_updated, edge_feats=cache.target_edge_features,
                        neg_sample_ratio=1)
                    total_memory_write_back_time += time.time() - memory_write_back_start_time

            model_train_start_time = time.time()
            loss = criterion(pred_pos, torch.ones_like(pred_pos))
            loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss) * num_target_nodes
            loss.backward()
            syn_model(model, 1)
            optimizer.step()
            total_model_train_time += time.time() - model_train_start_time

            cache_edge_ratio_sum += cache.cache_edge_ratio
            cache_node_ratio_sum += cache.cache_node_ratio
            total_samples += num_target_nodes
            i += 1
            epoch_time += time.time()-start_time

            if (i+1) % args.print_freq == 0 and args.node_rank == 0:
                val_ap, val_auc = evaluate(
                val_loader, sampler, model, criterion, cache, device, local_group)
                auc_list.append(val_auc)
                loss_list.append(total_loss)
                if len(tb_list) == 0:
                    tb_list.append(epoch_time)
                else:
                    tb_list.append(epoch_time+tb_list[-1])
                epoch_time = 0
                if args.distributed:
                    metrics = torch.tensor([val_ap, val_auc, total_loss, cache_edge_ratio_sum,
                                            cache_node_ratio_sum, total_samples,
                                            total_sampling_time, total_feature_fetch_time,
                                            total_memory_update_time,
                                            total_memory_write_back_time,
                                            total_model_train_time
                                            ]).to(device)
                    torch.distributed.all_reduce(metrics, group=local_group)
                    metrics /= args.local_world_size
                    val_ap, val_auc, total_loss, cache_edge_ratio_sum, cache_node_ratio_sum, \
                        total_samples, total_sampling_time, total_feature_fetch_time, \
                        total_memory_update_time, total_memory_write_back_time, \
                        total_model_train_time = metrics.tolist()

                if args.rank == 0:
                    logging.info('Epoch {:d}/{:d} | Iter {:d}/{:d} | Validation ap {:.4f} | Validation auc {:.4f} | Throughput {:.2f} samples/s | Loss {:.4f} | Cache node ratio {:.4f} | Cache edge ratio {:.4f} | Total Sampling Time {:.2f}s | Total Feature Fetching Time {:.2f}s | Total Memory Fetching Time {:.2f}s | Total Memory Update Time {:.2f}s | Total Memory Write Back Time {:.2f}s | Total Model Train Time {:.2f}s | Total Time {:.2f}s'.format(e + 1, args.epoch, i + 1, int(len(
                        train_loader)/args.local_world_size), val_ap, val_auc, total_samples * args.local_world_size / (time.time() - start_time), total_loss / (i + 1), cache_node_ratio_sum / (i + 1), cache_edge_ratio_sum / (i + 1), total_sampling_time, total_feature_fetch_time, total_memory_fetch_time, total_memory_update_time, total_memory_write_back_time, total_model_train_time, time.time() - start_time))

        # torch.distributed.barrier()
        
        epoch_time_sum += epoch_time

        # Validation
        if args.distributed and args.node_rank == 0:
            val_start = time.time()
            val_ap, val_auc = evaluate(
                val_loader, sampler, model, criterion, cache, device, local_group)
            
            val_end = time.time()
            val_time = val_end - val_start
    
        if args.distributed and args.node_rank == 0:
            metrics = torch.tensor([val_ap, val_auc, cache_edge_ratio_sum,
                                    cache_node_ratio_sum, total_samples,
                                    total_sampling_time, total_feature_fetch_time,
                                    total_memory_update_time,
                                    total_memory_write_back_time,
                                    total_model_train_time]).to(device)
            torch.distributed.all_reduce(metrics, group=local_group)
            metrics /= args.local_world_size
            val_ap, val_auc, cache_edge_ratio_sum, cache_node_ratio_sum, \
                total_samples, total_sampling_time, total_feature_fetch_time, \
                total_memory_update_time, total_memory_write_back_time, \
                total_model_train_time = metrics.tolist()

        if args.rank == 0:
            auc_list.append(val_auc)
            loss_list.append(total_loss)
            if len(tb_list) == 0:
                tb_list.append(epoch_time)
            else:
                tb_list.append(epoch_time+tb_list[-1])
            logging.info("Epoch {:d}/{:d} | train loss {:.4f} | Validation ap {:.4f} | Validation auc {:.4f} | Train time {:.2f} s | Validation time {:.2f} s | Train Throughput {:.2f} samples/s | Cache node ratio {:.4f} | Cache edge ratio {:.4f} | Total Sampling Time {:.2f}s | Total Feature Fetching Time {:.2f}s | Total Memory Fetching Time {:.2f}s | Total Memory Update Time {:.2f}s | Total Memory Write Back Time {:.2f}s | Total Model Train Time {:.2f}s".format(

                e + 1, args.epoch, total_loss, val_ap, val_auc, epoch_time, val_time, total_samples * args.local_world_size / epoch_time, cache_node_ratio_sum / (i + 1), cache_edge_ratio_sum / (i + 1), total_sampling_time, total_feature_fetch_time, total_memory_fetch_time, total_memory_update_time, total_memory_write_back_time, total_model_train_time))

        if args.rank == 0 and val_ap > best_ap:
            best_e = e + 1
            best_ap = val_ap
            model_to_save = model
            torch.save({
                'model': model_to_save.state_dict(),
                'memory': model_to_save.memory.backup() if args.use_memory else None
            }, checkpoint_path)
            logging.info(
                "Best val AP: {:.4f} & val AUC: {:.4f}".format(val_ap, val_auc))

        # if early_stopper.early_stop_check(val_ap):
        #     logging.info("Early stop at epoch {}".format(e))
        #     break
    
    for _ in range(int(length//args.num_nodes*(args.num_nodes-1))-int(length//args.num_nodes*args.node_rank)):
        optimizer.zero_grad()
        syn_model(model, 0)
        optimizer.step()

    if args.rank == 0:
        logging.info('Avg epoch time: {}'.format(epoch_time_sum / args.epoch))
        print(f'auc_list={auc_list}')
        print(f'loss_list={loss_list}')
        print(f'tb_list={tb_list}')

    if args.distributed:
        torch.distributed.barrier()

    if args.local_rank == 0:
        training = False
        gpu_load_thread.join()

    return best_e

def set_default_grad_to_zero(model):
    for param in model.parameters():
        if param.grad is None:
            param.grad = torch.zeros_like(param)

def syn_model(model, is_training: int):
    num_gpus = torch.tensor(data=(is_training), device=f'cuda:{args.local_rank}')
    dist.all_reduce(num_gpus, op=dist.ReduceOp.SUM)
    set_default_grad_to_zero(model)
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= num_gpus.item()
    pass

if __name__ == '__main__':
    main()
