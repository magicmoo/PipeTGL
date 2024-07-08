from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
# from torch.multiprocessing import Queue
from queue import Queue
from gnnflow.utils import mfgs_to_cuda
from modules.util import recv, recv_req
import torch.distributed as dist


import torch
import numpy as np


class FetchClient:
    def __init__(self, local_rank: int, global_rank, world_size: int, cnt_iterations: int, device, groups):
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
        self.cnt_interations = cnt_iterations
        self.device = device
        self.groups = groups
        self.next_data = None

        
        # if self.rank==0:
        #     self.pipe_status = create_shared_mem_array('pipe_status', [world_size*2], torch.int)
        # else:
        #     self.pipe_status = get_shared_mem_array('pipe_status', [world_size*2], torch.int)
        self.train_status = [0 for _ in range(5)]
        
        
    def start(self, q: Queue):
        iteration_now = self.rank
        model = self.model
        flag = False
        sends_thread1, sends_thread2 = None, None
        while True:
            if iteration_now + self.world_size >= self.cnt_interations:
                break
            param = q.get()
            if param is None:
                

            flag = True
            iteration_now += self.world_size


            
                        
            
        

def startFetchClient(client: FetchClient, q: Queue, q_input: Queue):
    client.start(q, q_input)