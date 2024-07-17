from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
# from torch.multiprocessing import Queue
from queue import Queue
from gnnflow.utils import mfgs_to_cuda
from modules.util import recv, recv_req
import torch.distributed as dist
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from modules.util import recv, send

import torch
import numpy as np


class SendClient:
    def __init__(self, local_rank: int, global_rank, local_world_size: int, world_size: int, cnt_iterations: int, device, model_data, groups):
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
        self.groups = groups
        self.model_data = model_data
        self.next_data = None

        
        # if self.rank==0:
        #     self.pipe_status = create_shared_mem_array('pipe_status', [world_size*2], torch.int)
        # else:
        #     self.pipe_status = get_shared_mem_array('pipe_status', [world_size*2], torch.int)
        self.train_status = [0 for _ in range(5)]
        
    def send_mem(self, mem, mail, dst, group, rank=-1):
        if mem is None:
            pass
        elif mem.shape[0] > 0:
            send([mem, mail], rank, dst, group)
        else:
            send(None, rank, dst, group)
    

    def start(self, q: Queue, q_syn: Queue):
        iteration_now = self.rank
        while True:
            if iteration_now + self.local_world_size >= self.cnt_interations:
                break
            mem, mail = q.get()
            # print('hello1', self.rank)
            # mem, mail = None, None
            dst = (self.local_rank+1)%self.local_world_size + self.local_world_size*self.node_rank
            group = self.groups[self.local_rank]
            self.send_mem(mem, mail, dst, group, self.rank)

            q_syn.put(None)
            # print('hello2', self.rank)

            params = q.get()
            group = self.groups[self.local_rank + self.local_world_size]

            if iteration_now+1+self.local_world_size != int(self.cnt_interations):
                send(params, self.rank, dst, group)
            else:
                for i, param in enumerate(params):
                    self.model_data[i][:] = param[:]
            
            q_syn.put(None)
            # print('hello3', self.rank)
            iteration_now += self.world_size


def startSendClient(client: SendClient, q: Queue, q_syn: Queue):
    client.start(q, q_syn)