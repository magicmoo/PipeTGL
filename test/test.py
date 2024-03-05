import torch
import numpy as np
from torch.multiprocessing import Process, Queue
import torch.distributed as dist
import time
import os
import threading

def send(tensors: list, target: int, group: object = None):
    ops = []
    for tensor in tensors:
        ops.append(dist.P2POp(dist.isend, tensor, target, group))
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

def recv(tensors: list, target: int, group: object = None):
    ops = []
    for tensor in tensors:
        ops.append(dist.P2POp(dist.irecv, tensor, target, group))
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()


def forward(iteration:int, idx: int):
    print(f'iteration={iteration}, rank={idx} forward begin at {time.perf_counter()-tb:.2f}')
    time.sleep(0.01)
    print(f'iteration={iteration}, rank={idx} forward already at {time.perf_counter()-tb:.2f}')

def predict(iteration:int, idx: int):
    print(f'iteration={iteration}, rank={idx} predict begin at {time.perf_counter()-tb:.2f}')
    time.sleep(0.05)
    print(f'iteration={iteration}, rank={idx} predict already at {time.perf_counter()-tb:.2f}')

def backward(iteration:int, idx: int):
    print(f'iteration={iteration}, rank={idx} backward begin at {time.perf_counter()-tb:.2f}')
    time.sleep(0.01)
    print(f'iteration={iteration}, rank={idx} backward already at {time.perf_counter()-tb:.2f}')

tb = 0
def train(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    flag = torch.empty(0).cuda()
    flag2 = torch.empty(0).cuda()

    dist.barrier()
    global tb
    tb = time.perf_counter()
    groups = []
    for i in range(world_size):
        g = dist.new_group([i, (i+1)%world_size])
        groups.append(g)
    for i in range(10):
        if rank==0:
            print(f'{rank}debug1, {i}')
            flag = torch.tensor([1+i, 2, 3]).cuda()
            flag2 = torch.tensor([1, 2, 3]).cuda()
            params = [flag.clone(), flag2.clone()]
            params2 = [flag, flag2]
            src = (rank-1+world_size)%world_size
            idx = src
            if i != 0:
                thread.join()
            if i != 0:
                recv(params2, src, groups[idx])
            print(f'{rank}debug2')
            dst = (rank+1)%world_size
            idx = (dst-1+world_size) % world_size
            thread = threading.Thread(target=send, args=(params, dst, groups[idx]))
            thread.start()
            print(f'{rank}debug3, {flag}, {flag2}')
        else:
            print(f'{rank}debug1, {i}')
            flag = torch.tensor([2+i, 2, 3]).cuda()
            flag2 = torch.tensor([2, 2, 3]).cuda()
            params = [flag.clone(), flag2.clone()]
            params2 = [flag, flag2]
            src = (rank-1+world_size)%world_size
            idx = src
            if i != 0:
                thread.join()
            recv(params2, src, groups[idx])
            print(f'{rank}debug2')
            dst = (rank+1)%world_size
            idx = (dst-1+world_size) % world_size
            if i != 9:
                thread = threading.Thread(target=send, args=(params, dst, groups[idx]))
                thread.start()
            print(f'{rank}debug3, {flag}, {flag2}')

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
world_size = 2
for i in range(world_size):
    p = Process(target=train, args=(i, world_size))
    p.start()