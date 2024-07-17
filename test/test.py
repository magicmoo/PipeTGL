import argparse
import logging
import math
import os
import random
import threading
import time
import sys
from venv import create
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

import GPUtil

import numpy as np
import torch
import torch.distributed as dist
import torch.nn
import torch.nn.parallel
import torch.utils.data
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import BatchSampler, SequentialSampler
from tqdm import tqdm
from torch.multiprocessing import Process, Queue
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
import torch.multiprocessing as mp

import gnnflow.cache as caches
from viztracer import VizTracer
from config import get_default_config
from gnnflow.data import (EdgePredictionDataset,
                          RandomStartBatchSampler, default_collate_ndarray)
from modules.tgnn import TGNN
from modules.sampler import DistributedBatchSampler

def worker(rank: int, world_size: int, queue: Queue):
    print(f'hello {rank}/{world_size}')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('nccl', world_size=world_size, rank=rank)
    t = torch.tensor([rank, rank], device=f'cuda:1')

    queue.put(t)
    # dist.barrier()
    print(f'sucess {rank}/{world_size}')

    time.sleep(3)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    queue = Queue()

    data = 10

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    process = mp.Process(target=worker, args=(rank+world_size, world_size*2, queue))
    process.start()

    dist.init_process_group('nccl', world_size=world_size*2, rank=rank)

    print(f'sucess {rank}/{world_size*2}')

    t = queue.get()
    print(t)

    # print(f"Result from worker process: {result}")



# import torch
# import torch.multiprocessing as mp

# def worker(queue):
#     # 从队列中获取 tensor
#     tensor = queue.get()
#     print("Received tensor:", tensor)

# def main():
#     # 创建一个队列
#     queue = mp.Queue()

#     # 创建一个 tensor
#     tensor = torch.tensor([1, 2, 3, 4, 5])

#     # 将 tensor 放入队列
#     queue.put(tensor)

#     # 创建并启动一个进程
#     process = mp.Process(target=worker, args=(queue,))
#     process.start()

#     # 等待进程结束
#     process.join()

# if __name__ == "__main__":
#     mp.set_start_method('spawn')  # 设置启动方式为 'spawn'
#     main()