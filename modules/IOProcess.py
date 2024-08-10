from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
# from torch.multiprocessing import Queue
from torch.multiprocessing import Process, Queue, Event
import queue
from gnnflow.utils import mfgs_to_cuda
from modules.util import recv, recv_req
import torch.distributed as dist


import torch
import numpy as np
import time

class IOProcess:
    def __init__(self, num_target_nodes, node_feats_shape, edge_feats_shape, local_rank: int, global_rank, local_world_size: int, world_size: int, cnt_iterations: int, num_epoch):
        """
        train_status[0] indicates whether the fetching feature has been completed
        train_status[1]: fetch memory for updating memory
        train_status[2]: fetch memory for training GNN
        train_status[3]: fetch model from last GPU
        train_status[4] indicates whether the training process has been completed
        """
        self.local_rank = local_rank
        self.rank = global_rank
        self.local_world_size = local_world_size
        self.world_size = world_size
        self.node_rank = self.rank//self.local_world_size
        self.num_nodes = self.world_size // self.local_world_size
        self.cnt_interations = cnt_iterations
        self.next_data = None
        self.num_target_nodes = num_target_nodes
        self.num_epoch = num_epoch
        if node_feats_shape is not None:
            self.node_feats = get_shared_mem_array('node_feats', node_feats_shape, torch.float32)
        else:
            self.node_feats = None
        if edge_feats_shape is not None:
            self.edge_feats = get_shared_mem_array('edge_feats', edge_feats_shape, torch.float32)
        else:
            self.edge_feats = None

        self.dim_node = 0 if self.node_feats is None else self.node_feats.shape[1]
        self.dim_edge = 0 if self.edge_feats is None else self.edge_feats.shape[1]
        if self.dim_node != 0:
            self.shm_nodes = get_shared_mem_array(f'nodes{self.local_rank}', [num_target_nodes*10+1], torch.int64)
            self.shm_node_feats = get_shared_mem_array(f'node_feats{self.local_rank}', [num_target_nodes*10, self.dim_node], torch.float32)
        if self.dim_edge != 0:
            self.shm_edges = get_shared_mem_array(f'edges{self.local_rank}', [num_target_nodes*10+1], torch.int64)
            self.shm_edge_feats = get_shared_mem_array(f'edge_feats{self.local_rank}', [num_target_nodes*10, self.dim_edge], torch.float32)
            self.shm_target_edges = get_shared_mem_array(f'target_edge{self.local_rank}', [num_target_nodes//3+1], torch.int64)
            self.shm_target_edge_feats = get_shared_mem_array(f'target_edge_feats{self.local_rank}', [num_target_nodes//3, self.dim_edge], torch.float32)

        
        # if self.rank==0:
        #     self.pipe_status = create_shared_mem_array('pipe_status', [world_size*2], torch.int)
        # else:
        #     self.pipe_status = get_shared_mem_array('pipe_status', [world_size*2], torch.int)
        self.train_status = [0 for _ in range(5)]

    def start(self, event1, event2):
        for _ in range(self.num_epoch):
            iteration_now = self.local_rank
            tt = 0
            while True:
                if iteration_now + self.local_world_size >= self.cnt_interations:
                    break
                event1.wait()
                event1.clear()
                if self.dim_node != 0:
                    num_nodes = self.shm_nodes[0]
                    nodes = self.shm_nodes[1:num_nodes+1]
                if self.dim_edge != 0:
                    num_edges, num_target_edges = self.shm_edges[0], self.shm_target_edges[0]
                    edges, eid = self.shm_edges[1:num_edges+1], self.shm_target_edges[1:num_target_edges+1]
                # nodes, edges, eid = q.get()
                t1 = time.time()
                if self.dim_node != 0:
                    self.shm_node_feats[:num_nodes] = self.node_feats[nodes]
                if self.dim_edge != 0:
                    self.shm_edge_feats[:num_edges] = self.edge_feats[edges]
                    self.shm_target_edge_feats[:num_target_edges] = self.edge_feats[eid]
                t2 = time.time()
                tt += t2-t1
                event2.set()

                iteration_now += self.local_world_size
            # print(tt)
        
def start_IOProcess(process, q, q_input):
    process.start(q, q_input)