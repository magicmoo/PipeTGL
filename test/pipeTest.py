import torch
import numpy as np
from torch.multiprocessing import Process, Queue
import torch.distributed as dist
import time
import os
import threading

def send(tensor: list, target: int, group = None):
    dist.send(tensor, target, group)


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

world_size = 2
tb = 0
def train(rank: int, world_size: int, q_input: Queue, q_output: Queue):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    flag = torch.empty(0).cuda()
    flag2 = torch.empty(0).cuda()
    dist.barrier()
    global tb
    tb = time.perf_counter()
    flag = True
    iterations = 10
    for iteration in range(iterations):
        if rank!=0 or iteration!=0:
            flag = torch.tensor([rank]).cuda()
            q_input.get()

        forward(iteration, rank)

        if rank+1 != world_size or iteration+1 != iterations:
            flag = torch.tensor([rank]).cuda()
            dst = (rank+1)%world_size
            q_output.put(flag)

        predict(iteration, rank)

        if rank!=0 or iteration!=0:
            # flag2 = torch.tensor([rank+1, rank+1]).cuda()
            src = (rank-1+world_size)%world_size
            # idx = src + world_size
            dist.recv(flag2, src)

        backward(iteration, rank)

        if rank+1 != world_size or iteration+1 != iterations:
            # flag2 = torch.tensor([rank+1, rank+1]).cuda()
            dst = (rank+1)%world_size
            idx = (dst-1+world_size)%world_size + world_size
            send_thread = threading.Thread(target=send, args=(flag2, dst))
            send_thread.start()

if __name__ == '__main__':
    queues = []
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    for i in range(world_size):
        q = Queue()
        queues.append(q)
    for i in range(world_size):
        p = Process(target=train, args=(i, world_size, queues[i], queues[(i+1)%world_size]))
        p.start()
    