import logging
import os
import random
import time
import gc
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from dgl.heterograph import DGLBlock
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array


def push_model(model, model_data):
    i = 0
    for param in model.parameters():
        model_data[i][:] = param.data[:].to('cpu')
        i += 1

def pull_model(model, model_data, device):
    i = 0
    for param in model.parameters():
        param.data[:] = model_data[i][:].to(device)
        i += 1

def send(tensors: list, rank: int, target: int, group: object = None):
    
    if tensors is None:
        tensor = torch.tensor([rank]).to(f'cuda:{rank}')
        # print(f'send1: {rank} {tensor}')
        req = dist.isend(tensor, target, group)
        # req.wait()
        # print(f'send1 finished: {rank}')
    else:
        # print(f'send2: {rank}')
        ops = []
        for tensor in tensors:
            ops.append(dist.P2POp(dist.isend, tensor, target, group))
        reqs = dist.batch_isend_irecv(ops)
        # for req in reqs:
        #     req.wait()
        # print(f'send2 finished: {rank}')
    
    
def recv(tensors: list, rank: int, target: int, group: object = None):
    if tensors is None:
        # print(f'recv1: {rank} from {target}')
        tensor = torch.tensor([rank]).to(f'cuda:{rank}')
        req = dist.irecv(tensor, target, group)
        req.wait()
        # print(f'recv1 finished: {rank}, {tensor}')
        if tensor.sum() == target:
            return True
        else:
            return False
    else:
        # print(f'recv2: {rank} from {target}')
        ops = []
        for tensor in tensors:
            ops.append(dist.P2POp(dist.irecv, tensor, target, group))
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        # print(f'recv2 finished: {rank}')

def recv_req(tensors: list, rank: int, target: int, group: object = None):
    # determine that tensors is a list

    # print(f'recv2: {rank} from {target}')
    ops = []
    for tensor in tensors:
        ops.append(dist.P2POp(dist.irecv, tensor, target, group))
    reqs = dist.batch_isend_irecv(ops)
    # print(f'recv2 finished: {rank}')
    return reqs
    

def load_feat(dataset: str, data_dir: Optional[str] = None,
              shared_memory: bool = False, local_rank: int = 0, rank: int = 0, local_world_size: int = 1,
              memmap: bool = False, load_node: bool = True, load_edge: bool = True, local_group=None):
    """
    Loads the node and edge features for the given dataset.

    NB: either node_feats or edge_feats can be None, but not both.

    Args:
        dataset: the name of the dataset.
        data_dir: the directory where the dataset is stored.
        shared_memory: whether to use shared memory.
        local_rank: the local rank of the process.
        local_world_size: the local world size of the process.
        memmap (bool): whether to use memmap.
        load_node (bool): whether to load node features.
        load_edge (bool): whether to load edge features.

    Returns:
        node_feats: the node features. (None if not available)
        edge_feats: the edge features. (None if not available)
    """

    dataset_path = os.path.join(data_dir, dataset)
    node_feat_path = os.path.join(dataset_path, 'node_features.npy')
    edge_feat_path = os.path.join(dataset_path, 'edge_features.npy')

    if not os.path.exists(node_feat_path) and \
            not os.path.exists(edge_feat_path):
        raise ValueError("Both {} and {} do not exist".format(
            node_feat_path, edge_feat_path))

    mmap_mode = "r+" if memmap else None

    node_feats = None
    edge_feats = None
    if not shared_memory or (shared_memory and local_rank == 0):
        if os.path.exists(node_feat_path) and load_node:
            node_feats = np.load(
                node_feat_path, mmap_mode=mmap_mode, allow_pickle=False)
            if not memmap:
                node_feats = torch.from_numpy(node_feats)

        if os.path.exists(edge_feat_path) and load_edge:
            edge_feats = np.load(
                edge_feat_path, mmap_mode=mmap_mode, allow_pickle=False)
            if not memmap:
                edge_feats = torch.from_numpy(edge_feats)

    if shared_memory:
        node_feats_shm, edge_feats_shm = None, None
        if local_rank == 0:
            if node_feats is not None:
                node_feats = node_feats.to(torch.float32)
                node_feats_shm = create_shared_mem_array(
                    'node_feats', node_feats.shape, node_feats.dtype)
                node_feats_shm[:] = node_feats[:]
            if edge_feats is not None:
                edge_feats = edge_feats.to(torch.float32)
                edge_feats_shm = create_shared_mem_array(
                    'edge_feats', edge_feats.shape, edge_feats.dtype)
                edge_feats_shm[:] = edge_feats[:]
            # broadcast the shape and dtype of the features
            node_feats_shape = node_feats.shape if node_feats is not None else None
            edge_feats_shape = edge_feats.shape if edge_feats is not None else None
            shapes = [node_feats_shape, edge_feats_shape]
            # dist.barrier()
                    
            torch.distributed.broadcast_object_list(
                shapes, src=0)

        if local_rank != 0:
            # dist.barrier()
            shapes = [None, None]
            torch.distributed.broadcast_object_list(
                shapes, src=0)
            node_feats_shape, edge_feats_shape = shapes
            if node_feats_shape is not None:
                node_feats_shm = get_shared_mem_array(
                    'node_feats', node_feats_shape, torch.float32)
            if edge_feats_shape is not None:
                edge_feats_shm = get_shared_mem_array(
                    'edge_feats', edge_feats_shape, torch.float32)

        torch.distributed.barrier()
        if node_feats_shm is not None:
            logging.info("rank {} node_feats_shm shape {}".format(
                local_rank, node_feats_shm.shape))

        if edge_feats_shm is not None:
            logging.info("rank {} edge_feats_shm shape {}".format(
                local_rank, edge_feats_shm.shape))

        return node_feats_shm, edge_feats_shm

    return node_feats, edge_feats


def save_memory(model):
    node_memory = model.memory.node_memory.clone()
    node_memory_ts = model.memory.node_memory_ts.clone()
    mailbox = model.memory.mailbox.clone()
    mailbox_ts = model.memory.mailbox_ts.clone()
    return (node_memory, node_memory_ts, mailbox, mailbox_ts)


def put_memory(input, model):
    node_memory, node_memory_ts, mailbox, mailbox_ts = input
    model.memory.node_memory[:] = node_memory[:]
    model.memory.node_memory_ts[:] = node_memory_ts[:]
    model.memory.mailbox[:] = mailbox[:]
    model.memory.mailbox_ts[:] = mailbox_ts[:]