import torch
import torch.distributed as dist
import os
import datetime
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from modules.util import load_feat

print('hello')
if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # ip = os.environ['MASTER_ADDR']
    # print(ip)
    os.environ['MASTER_ADDR'] = '192.168.122.101'
    os.environ['MASTER_PORT'] = '29500'
    print(f'debug {rank}/{world_size}')
    dist.init_process_group('nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    # torch.cuda.set_device(local_rank)
    print(f'{rank}/{world_size}, {local_rank}')

    dist.barrier()

    if rank==0:
        shapes = [torch.tensor(123), torch.tensor(321)]
        torch.distributed.broadcast_object_list(
                shapes, src=0)
    else:
        shapes = [None, None]
        torch.distributed.broadcast_object_list(
                shapes, src=0)
    print(shapes)

    # node_feats, edge_feats = load_feat(
    #     'REDDIT', data_dir='/data/TGL', shared_memory=True,
    #     local_rank=local_rank, local_world_size=local_world_size)
    