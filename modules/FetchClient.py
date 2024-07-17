from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
# from torch.multiprocessing import Queue
from queue import Queue
from gnnflow.utils import mfgs_to_cuda
from modules.util import recv, recv_req
import torch.distributed as dist


import torch
import numpy as np


class FetchClient:
    def __init__(self, local_rank: int, global_rank, local_world_size: int, world_size: int, cnt_iterations: int, model, device, cache, model_data, groups):
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
        self.next_data = None

        
        # if self.rank==0:
        #     self.pipe_status = create_shared_mem_array('pipe_status', [world_size*2], torch.int)
        # else:
        #     self.pipe_status = get_shared_mem_array('pipe_status', [world_size*2], torch.int)
        self.train_status = [0 for _ in range(5)]
        
        
    def start(self, q: Queue, q_input: Queue, q_syn: Queue):
        iteration_now = self.local_rank
        model = self.model
        flag = False
        sends_thread1, sends_thread2 = None, None
        stream = torch.cuda.Stream()
        while True:
            if iteration_now + self.local_world_size >= self.cnt_interations:
                break
            stream1 = torch.cuda.Stream()
            with torch.cuda.stream(stream1):
                mfgs, eid = q_input.get()
                mfgs_to_cuda(mfgs, self.device)
                mfgs[0][0].edata['dt'] = mfgs[0][0].edata['dt'].to(self.device)
                mfgs = self.cache.fetch_feature(
                    mfgs, eid)
            stream1.synchronize()
            q.put(mfgs)

            # if sends_thread1 is not None:
            #     sends_thread1.join()
            if iteration_now >= self.local_world_size:
                q_syn.get()
            # print('fetch ', self.rank)
            stream2 = torch.cuda.Stream()
            with torch.cuda.stream(stream2):
                idx = (self.local_rank-1+self.local_world_size)%self.local_world_size
                src = (self.local_rank-1+self.local_world_size)%self.local_world_size + self.node_rank*self.local_world_size
                mem, mail = model.memory.recv_mem(iteration_now, self.local_rank, self.local_world_size, self.device, self.groups[idx], src=src)
            stream2.synchronize()
            q.put((mem, mail))
            
            stream3 = torch.cuda.Stream()
            with torch.cuda.stream(stream3):
                b = mfgs[0][0]
                length = len(eid)*2
                overlap_nid = b.srcdata['ID'][:length]
                all_nodes = b.srcdata['ID']
                overlap_nid = torch.unique(overlap_nid).cpu()
                all_nodes = torch.unique(all_nodes).cpu()
                pull_nodes = torch.from_numpy(np.setdiff1d(all_nodes.numpy(), overlap_nid.numpy()))
                mem = model.memory.node_memory[pull_nodes].to(self.device, non_blocking=True)
                mem_ts = model.memory.node_memory_ts[pull_nodes].to(self.device, non_blocking=True)
                mail = model.memory.mailbox[pull_nodes].to(self.device, non_blocking=True)
                mail_ts = model.memory.mailbox_ts[pull_nodes].to(self.device, non_blocking=True)

            stream3.synchronize()
            q.put((pull_nodes, mem, mem_ts, mail, mail_ts))
    
            # if sends_thread2 is not None:
            #     sends_thread2.join()
            # if iteration_now >= self.local_world_size:
            q_syn.get()

            stream4 = torch.cuda.Stream()
            with torch.cuda.stream(stream4):
                params = [torch.zeros_like(param.data, device=self.device) for param in model.parameters()]
                if self.local_rank!=0 or flag:
                    src = (self.rank-1+self.world_size)%self.world_size
                    idx = src + self.world_size
                    recv(params, self.rank, src, self.groups[idx])
                else:
                    for i, param in enumerate(params):
                        param[:] = self.model_data[i][:].to(self.device)
            stream4.synchronize()
            q.put(params)

            flag = True
            iteration_now += self.local_world_size

def startFetchClient(client: FetchClient, q: Queue, q_input: Queue, q_syn: Queue):
    client.start(q, q_input, q_syn)