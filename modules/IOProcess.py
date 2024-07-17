from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
# from torch.multiprocessing import Queue
from torch.multiprocessing import Process, Queue
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
        
    def start_fetch_and_send():
        