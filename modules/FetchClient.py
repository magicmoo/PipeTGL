from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
# from torch.multiprocessing import Queue
from queue import Queue
from gnnflow.utils import mfgs_to_cuda
from modules.util import recv
import torch.distributed as dist


import torch
import numpy as np


class FetchClient:
    def __init__(self, local_rank: int, global_rank, world_size: int, model, device, sampler, cache, model_data, groups):
        """
        train_status[0] indicates whether the fetching feature has been completed
        train_status[1]: fetch memory for updating memory
        train_status[2]: fetch memory for training GNN
        train_status[3]: fetch model from last GPU
        train_status[4] indicates whether the training process has been completed
        """
        self.rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.sampler = sampler
        self.device = device
        self.cache = cache
        self.groups = groups
        self.model = model
        self.model_data = model_data
        self.next_data = None

        
        # if self.rank==0:
        #     self.pipe_status = create_shared_mem_array('pipe_status', [world_size*2], torch.int)
        # else:
        #     self.pipe_status = get_shared_mem_array('pipe_status', [world_size*2], torch.int)
        self.train_status = [0 for _ in range(5)]
        
        
    def start(self, q: Queue, q_input: Queue):
        iteration_now = self.rank
        model = self.model
        flag = False
        while True:
            # while q_input.qsize() == 0:
            #     if self.train_status[4] == 1:
            #         break
            #     pass
            # if self.train_status[4] == 1:
            #     break
            target_nodes, ts, eid = q_input.get(timeout=3)
            mfgs = self.sampler.sample(target_nodes, ts)
            mfgs_to_cuda(mfgs, self.device)
            mfgs = self.cache.fetch_feature(
                mfgs, eid)
            # while self.train_status[0] == 1:
            #     pass
            q.put(mfgs)
            self.train_status[0] = 1
            
            # while self.train_status[1] == 1:
            #     pass
            if flag:
                sends_thread1 = q_input.get()
                sends_thread1.join()
            idx = (self.rank-1+self.world_size)%self.world_size
            mem, mail = model.memory.recv_mem(iteration_now, self.rank, self.world_size, self.device, self.groups[idx])
            q.put((mem, mail))
            self.train_status[1] = 1
            
            b = mfgs[0][0]
            length = len(eid)*2
            overlap_nid = b.srcdata['ID'][:length]
            all_nodes = b.srcdata['ID']
            overlap_nid = torch.unique(overlap_nid).cpu()
            all_nodes = torch.unique(all_nodes).cpu()
            pull_nodes = torch.from_numpy(np.setdiff1d(all_nodes.numpy(), overlap_nid.numpy()))
            mem = model.memory.node_memory[pull_nodes].to(self.device)
            mem_ts = model.memory.node_memory_ts[pull_nodes].to(self.device)
            mail = model.memory.mailbox[pull_nodes].to(self.device)
            mail_ts = model.memory.mailbox_ts[pull_nodes].to(self.device)
            # while self.train_status[2] == 1:
            #     pass
            q.put((mem, mem_ts, mail, mail_ts))
            self.train_status[2] = 1
            
            # while self.train_status[3] == 1:
            #     pass
            if flag:
                sends_thread2 = q_input.get()
                sends_thread2.join()
            params = [torch.zeros_like(param.data, device=self.device) for param in model.parameters()]
            if self.rank!=0 or flag:
                src = (self.rank-1+self.world_size)%self.world_size
                idx = src + self.world_size
                recv(params, self.global_rank, src, self.groups[idx])
            else:
                for i, param in enumerate(params):
                    param[:] = self.model_data[i][:].to(self.device)
            q.put(params)
            self.train_status[3] = 1
            flag = True
            iteration_now += self.world_size

            
        self.train_status[4] = 1
            
                        
            
        

def startFetchClient(client: FetchClient, q: Queue, q_input: Queue):
    client.start(q, q_input)