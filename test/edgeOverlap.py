import argparse
from ctypes.wintypes import tagRECT
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

import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from modules.sampler import DistributedBatchSampler
import gnnflow.cache as caches
from config import get_default_config
from gnnflow.data import (EdgePredictionDataset,
                          RandomStartBatchSampler, default_collate_ndarray)
# import sys
# path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(path)
# from modules.sampler import DistributedBatchSampler
# from modules.sampler import DistributedBatchSampler
from gnnflow.models.dgnn import DGNN
from gnnflow.models.gat import GAT
from gnnflow.models.graphsage import SAGE
from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import (DstRandEdgeSampler, EarlyStopMonitor,
                           build_dynamic_graph, get_pinned_buffers,
                           get_project_root_dir, load_dataset, load_feat,
                           mfgs_to_cuda)
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = ['REDDIT', 'GDELT', 'LASTFM', 'MAG', 'MOOC', 'WIKI']
model_names = ['TGN', 'TGAT', 'DySAT', 'GRAPHSAGE', 'GAT']
cache_names = sorted(name for name in caches.__dict__
                     if not name.startswith("__")
                     and callable(caches.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=model_names, required=True,
                    help="model architecture" + '|'.join(model_names))
parser.add_argument("--data", choices=datasets, required=True,
                    help="dataset:" + '|'.join(datasets))
parser.add_argument("--epoch", help="maximum training epoch",
                    type=int, default=50)
parser.add_argument("--lr", help='learning rate', type=float, default=0.0001)
parser.add_argument("--num-workers", help="num workers for dataloaders",
                    type=int, default=1)
parser.add_argument("--num-chunks", help="number of chunks for batch sampler",
                    type=int, default=1)
parser.add_argument("--print-freq", help="print frequency",
                    type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ingestion-batch-size", type=int, default=10000000,
                    help="ingestion batch size")

# optimization
parser.add_argument("--cache", choices=cache_names, help="feature cache:" +
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

def main():
    args.distributed = int(os.environ.get('WORLD_SIZE', 0)) > 1
    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group('nccl')
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        args.local_rank = args.rank = 0
        args.local_world_size = args.world_size = 1

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

    data_path = os.path.join(path, 'data')
    train_data, val_data, test_data, full_data = load_dataset(args.data, data_dir=data_path)
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
            drop_last=False, rank=args.rank, world_size=args.world_size,
            num_chunks=args.num_chunks)
        val_sampler = DistributedBatchSampler(
            SequentialSampler(val_ds),
            batch_size=batch_size, drop_last=False, rank=args.rank,
            world_size=args.world_size)
        test_sampler = DistributedBatchSampler(
            SequentialSampler(test_ds),
            batch_size=batch_size, drop_last=False, rank=args.rank,
            world_size=args.world_size)
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
        args.data, shared_memory=args.distributed, data_dir=data_path,
        local_rank=args.local_rank, local_world_size=args.local_world_size)

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

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True)

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
                   model, optimizer, criterion, cache, device)

    logging.info('Loading model at epoch {}...'.format(best_e))
    ckpt = torch.load(checkpoint_path)
    if args.distributed:
        model.module.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt['model'])
    if args.use_memory:
        if args.distributed:
            model.module.memory.restore(ckpt['memory'])
        else:
            model.memory.restore(ckpt['memory'])
    if args.distributed:
        metrics = torch.tensor([ap, auc], device=device)
        torch.distributed.all_reduce(metrics)
        metrics /= args.world_size
        ap, auc = metrics.tolist()
        if args.rank == 0:
            logging.info('Test ap:{:4f}  test auc:{:4f}'.format(ap, auc))

    if args.distributed:
        torch.distributed.barrier()


def train(train_loader, val_loader, sampler, model, optimizer, criterion,
          cache, device):
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

    logging.info('Start training...')
    if args.distributed:
        torch.distributed.barrier()

    for e in range(args.epoch):
        model.train()
        cache.reset()
        if e > 0:
            if args.distributed:
                model.module.reset()
            else:
                model.reset()
        train_iter = iter(train_loader)
        target_nodes, ts, eid = next(train_iter)
        mfgs = sampler.sample(target_nodes, ts)
        next_data = (mfgs, eid)
        last_nodes = None

        sampling_thread = None

        i = 0
        while True:
            if sampling_thread is not None:
                sampling_thread.join()

            mfgs, eid = next_data
            num_target_nodes = len(eid) * 2
            target_nodes = mfgs[0][0].srcdata['ID'][:num_target_nodes]
            target_nodes = torch.unique(target_nodes)
            target_nodes = set(target_nodes.tolist())
            # print(target_nodes)
            if last_nodes != None:
                print(len(last_nodes.intersection(target_nodes))/len(target_nodes))
            last_nodes = target_nodes

            # Sampling for next batch
            try:
                next_target_nodes, next_ts, next_eid = next(train_iter)
            except StopIteration:
                break
            sampling_thread = threading.Thread(target=sampling, args=(
                next_target_nodes, next_ts, next_eid))
            sampling_thread.start()


    return best_e


if __name__ == '__main__':
    main()
