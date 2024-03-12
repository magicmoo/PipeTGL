import torch
import numpy as np
from torch.multiprocessing import Process, Queue
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
import torch.distributed as dist
import time
import os
import threading

tb = 0

dist.init_process_group('nccl')
rank = dist.get_rank()
if rank == 0:
    mem = create_shared_mem_array('mem', (2, 2), torch.int)
    dist.barrier()
    mem[:] = 2
    dist.barrier()
    dist.barrier()
    print(mem, '0')
else:
    dist.barrier()
    mem = get_shared_mem_array('mem', (2, 2), torch.int)
    dist.barrier()
    print(mem, '1')
    mem[:] = 1
