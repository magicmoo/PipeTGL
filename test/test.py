import torch
import torch.distributed as dist
import time

dist.init_process_group('nccl')

rank = dist.get_rank()

x = torch.tensor([rank for _ in range(100000000)])
dist.barrier()
t1 = time.time()
y = x.to(f'cuda:{rank}')
t2 = time.time()
print(t2-t1)
