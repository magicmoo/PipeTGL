from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
import torch.distributed as dist
import torch

class SynClass:
    def __init__(self, rank:int, world_size:int, syn_iterations:int):
        self.rank = rank
        self.world_size = world_size
        self.syn_itertaions = syn_iterations
        if rank == 0:
            self.syn_data = create_shared_mem_array('syn', [1], torch.int)
            dist.barrier()
        elif rank+1 == world_size:
            dist.barrier()
            self.syn_data = get_shared_mem_array('syn', [1], torch.int)
        else:
            dist.barrier()
            self.syn_data = None
    def syn(self, iteration_now: int):
        if self.rank == 0 and (iteration_now//self.world_size)%self.syn_itertaions == 0:
            while self.syn_data[0] != 1:
                pass
            self.syn_data[0] = 0
            return True
        elif self.rank+1 == self.world_size and (iteration_now//self.world_size + 1)%self.syn_itertaions == 0:
            self.syn_data[0] = 1
        return False
        