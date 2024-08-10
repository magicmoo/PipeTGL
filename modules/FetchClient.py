from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
# from torch.multiprocessing import Queue
from gnnflow.utils import mfgs_to_cuda
from modules.util import recv, recv_req
import torch.distributed as dist
from torch.multiprocessing import Event, Process
import torch.multiprocessing as mp
from queue import Queue
import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from modules.IOProcess import start_IOProcess, IOProcess


import torch
import numpy as np
import time


class FetchClient:
    def __init__(self, local_rank: int, global_rank, local_world_size: int, world_size: int, cnt_iterations: int, model, device, cache, model_data, groups, num_target_nodes, node_feats, edge_feats, num_epoch):
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
        self.device = device
        self.cache = cache
        self.groups = groups
        self.model = model
        self.model_data = model_data
        self.stream = torch.cuda.Stream(device)

        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.dim_node = 0 if node_feats is None else node_feats.shape[1]
        self.dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
        if self.dim_node != 0:
            self.shm_nodes = create_shared_mem_array(f'nodes{self.local_rank}', [num_target_nodes*10+1], torch.int64)
            self.shm_node_feats = create_shared_mem_array(f'node_feats{self.local_rank}', [num_target_nodes*10, self.dim_node], torch.float32)
        if self.dim_edge != 0:
            self.shm_edges = create_shared_mem_array(f'edges{self.local_rank}', [num_target_nodes*10+1], torch.int64)
            self.shm_edge_feats = create_shared_mem_array(f'edge_feats{self.local_rank}', [num_target_nodes*10, self.dim_edge], torch.float32)
            self.shm_target_edges = create_shared_mem_array(f'target_edge{self.local_rank}', [num_target_nodes//3+1], torch.int64)
            self.shm_target_edge_feats = create_shared_mem_array(f'target_edge_feats{self.local_rank}', [num_target_nodes//3, self.dim_edge], torch.float32)
        node_feats_shape = None if node_feats is None else node_feats.shape
        edge_feats_shape = None if edge_feats is None else edge_feats.shape
        self.event1, self.event2 = Event(), Event()
        self.num_epoch = num_epoch
        self.IOClient = IOProcess(num_target_nodes, node_feats_shape, edge_feats_shape, local_rank, global_rank, local_world_size, world_size, cnt_iterations, num_epoch)
        
        self.next_data = None


        # if self.rank==0:
        #     self.pipe_status = create_shared_mem_array('pipe_status', [world_size*2], torch.int)
        # else:
        #     self.pipe_status = get_shared_mem_array('pipe_status', [world_size*2], torch.int)
        self.train_status = [0 for _ in range(5)]
    
    
    def start(self, q: Queue, q_input: Queue, q_syn: Queue):
        for _ in range(self.num_epoch):
            iteration_now = self.local_rank
            model = self.model
            flag = False
            sends_thread1, sends_thread2 = None, None
            stream = self.stream
            params = [torch.zeros_like(param.data, device=self.device) for param in model.parameters()]
            src = (self.rank-1+self.world_size)%self.world_size
            
            event1, event2 = self.event1, self.event2

            tt = 0
            while True:
                if iteration_now + self.local_world_size >= self.cnt_interations:
                    break
                stream.wait_stream(torch.cuda.current_stream())
                # with torch.cuda.stream(stream):
                    # mfgs, eid = q_input.get()
                    # nodes, edges = mfgs[0][0].srcdata['ID'], mfgs[0][0].edata['ID']
                    # t1 = time.time()
                    # # q_IO1.put((nodes, edges, eid))
                    # if self.dim_node != 0:
                    #     num_nodes = nodes.shape[0]
                    #     self.shm_nodes[0] = num_nodes
                    #     self.shm_nodes[1:num_nodes+1] = nodes[:]
                    # if self.dim_edge != 0:
                    #     num_edges = edges.shape[0]
                    #     self.shm_edges[0] = num_edges
                    #     num_target_edges = len(eid)
                    #     self.shm_target_edges[0] = num_target_edges
                    #     self.shm_edges[1:num_edges+1], self.shm_target_edges[1:num_target_edges+1] = edges[:], torch.tensor(eid)
                    # stream.synchronize()
                    # event1.set()
                    # t2 = time.time()
                    # tt += t2-t1
                    # event2.wait()
                    # event2.clear()
                    # t1 = time.time()
                    # mfgs_to_cuda(mfgs, self.device)
                    # if self.dim_node != 0:
                    #     for b in mfgs[0]:
                    #         nodes = b.srcdata['ID']
                    #         b.srcdata['h'] = self.shm_node_feats[:nodes.shape[0]].to(self.device, non_blocking=True)
                    # if self.dim_edge != 0:
                    #     for mfg in mfgs:
                    #         for b in mfg:
                    #             edges = b.edata['ID']
                    #             if len(edges) == 0:
                    #                 continue
                            
                    #             b.edata['f'] = self.shm_edge_feats[:edges.shape[0]].to(self.device, non_blocking=True)
                    #     self.cache.target_edge_features = self.shm_target_edge_feats[:len(eid)].to(self.device, non_blocking=True)
                    # stream.synchronize()
                    # q.put(mfgs)
                #     t2 = time.time()
                #     tt += t2-t1
                    

                # if sends_thread1 is not None:
                #     sends_thread1.join()
                # if iteration_now >= self.local_world_size:
                #     q_syn.get()
                #     q_syn.get()
                # # print('fetch ', self.rank)
                # stream.wait_stream(torch.cuda.current_stream()) 
                # with torch.cuda.stream(stream):
                #     idx = (self.local_rank-1+self.local_world_size)%self.local_world_size
                #     mem, mail = model.memory.recv_mem(iteration_now, self.local_rank, self.local_world_size, self.device, self.groups[idx], src=src)
                #     stream.synchronize()
                #     q.put((mem, mail))
                
                # stream.wait_stream(torch.cuda.current_stream())
                # with torch.cuda.stream(stream):
                #     b = mfgs[0][0]
                #     length = len(eid)*2
                #     overlap_nid = b.srcdata['ID'][:length]
                #     all_nodes = b.srcdata['ID']
                #     overlap_nid = torch.unique(overlap_nid).cpu()
                #     all_nodes = torch.unique(all_nodes).cpu()
                #     pull_nodes = torch.from_numpy(np.setdiff1d(all_nodes.numpy(), overlap_nid.numpy()))
                #     mem = model.memory.node_memory[pull_nodes].to(self.device)
                #     mem_ts = model.memory.node_memory_ts[pull_nodes].to(self.device)
                #     mail = model.memory.mailbox[pull_nodes].to(self.device)
                #     mail_ts = model.memory.mailbox_ts[pull_nodes].to(self.device)
                #     stream.synchronize()
                #     q.put((pull_nodes, mem, mem_ts, mail, mail_ts))
        
                # if sends_thread2 is not None:
                #     sends_thread2.join()
                # if iteration_now >= self.local_world_size:
                
                # q_syn.get()
                # stream.wait_stream(torch.cuda.current_stream())
                # with torch.cuda.stream(stream):
                #     if self.local_rank!=0 or flag:
                #         idx = src + self.world_size
                #         recv(params, self.rank, src, self.groups[idx])
                #     else:
                #         for i, param in enumerate(params):
                #             param[:] = self.model_data[i][:].to(self.device)
                    
                #     stream.synchronize()
                #     q.put(params)

                # flag = True
                iteration_now += self.local_world_size
            # print(tt)

def startFetchClient(client: FetchClient, q: Queue, q_input: Queue, q_syn: Queue):
    client.start(q, q_input, q_syn)