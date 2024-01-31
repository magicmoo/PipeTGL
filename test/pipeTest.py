import torch
import numpy as np
from torch.multiprocessing import Process, Queue
import torch.distributed as dist
import time
import os

def forward(iteration:int, idx: int):
    print(f'iteration={iteration}, rank={idx} forward begin at {time.perf_counter()-tb:.2f}')
    time.sleep(0.01)
    print(f'iteration={iteration}, rank={idx} forward already at {time.perf_counter()-tb:.2f}')

def predict(iteration:int, idx: int):
    print(f'iteration={iteration}, rank={idx} predict begin at {time.perf_counter()-tb:.2f}')
    time.sleep(0.04)
    print(f'iteration={iteration}, rank={idx} predict already at {time.perf_counter()-tb:.2f}')

def backward(iteration:int, idx: int):
    print(f'iteration={iteration}, rank={idx} backward begin at {time.perf_counter()-tb:.2f}')
    time.sleep(0.04)
    print(f'iteration={iteration}, rank={idx} backward already at {time.perf_counter()-tb:.2f}')

dist.init_process_group(backend='nccl')
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
flag = torch.empty(0).to(f'cuda:{rank}')

dist.barrier()
tb = time.perf_counter()
for iteration in range(100):

    if rank!=0 or iteration!=0:
        request = dist.irecv(flag, (rank-1+world_size)%world_size)
        request.wait()

    forward(iteration, rank)

    dist.isend(flag, (rank+1)%world_size)

    predict(iteration, rank)

    if rank!=0 or iteration!=0:
        request = dist.irecv(flag, (rank-1+world_size)%world_size)
        request.wait()

    backward(iteration, rank)

    dist.isend(flag, (rank+1)%world_size)