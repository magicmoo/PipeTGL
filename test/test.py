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
torch.cuda.set_device(rank)
flag = torch.empty(0).to(f'cuda:{rank}')

dist.barrier()
tb = time.perf_counter()
iterations = 1
for iteration in range(iterations):

    if rank!=0 or iteration!=0:
        request = dist.irecv(flag, (rank-1+world_size)%world_size)
        request.wait()

    forward(iteration, rank)
    
    if rank!=world_size-1 or iteration!=iterations-1:
        dist.isend(flag, (rank+1)%world_size)
    
    predict(iteration, rank)

    p1 = torch.zeros((2, 2)).cuda()+torch.ones((2, 2)).cuda()*rank
    p2 = torch.ones((2, 2)).cuda()+torch.ones((2, 2)).cuda()*rank
    params = [p1.requires_grad_(True), p2.requires_grad_(True)]

    y = params[0]*(rank+1) + params[1]*(rank+2)
    y.sum().backward()
    # print(pa.grad)

    print()

    if rank!=0 or iteration!=0:
        print(params)
        print([param.grad for param in params])

        ops = []
        for param in params:
            ops.append(dist.P2POp(dist.irecv, param, (rank-1+world_size)%world_size))
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        
        print(params)
        print([param.grad for param in params])

    backward(iteration, rank)
    if rank!=world_size-1 or iteration!=iterations-1:
        ops = []
        for param in params:
            ops.append(dist.P2POp(dist.isend, param, (rank+1)%world_size))
        dist.batch_isend_irecv(ops)


# x = torch.randn((2, 2), requires_grad=True)
# y = 2*x
# y.sum().backward()
# print(x.grad)