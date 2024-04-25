from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
import time
import numpy as np
from dgl.heterograph import DGLBlock
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from gnnflow.distributed.kvstore import KVStoreClient
import gnnflow.utils as utils
from modules import memory_updater
from modules.util import recv, send, recv_req
import threading


class Memory:
    """
    Memory module proposed by TGN
    """

    def __init__(self, num_nodes: int, dim_edge: int, dim_memory: int,
                 device: Union[torch.device, str] = 'cpu',
                 shared_memory: bool = False,
                 kvstore_client: Optional[KVStoreClient] = None):
        """
        Args:
            num_nodes: number of nodes in the graph
            dim_edge: dimension of the edge features
            dim_time: dimension of the time encoding
            dim_memory: dimension of the output of the memory
            device: device to store the memory
            shared_memory: whether to store in shared memory (for multi-GPU training)
            kvstore_client: The KVStore_Client for fetching memorys when using partition
        """
        if shared_memory:
            device = 'cpu'

        self.num_nodes = num_nodes
        self.dim_edge = dim_edge
        self.dim_memory = dim_memory
        # raw message: (src_memory, dst_memory, edge_feat)
        self.dim_raw_message = 2 * dim_memory + dim_edge
        self.device = device

        self.kvstore_client = kvstore_client
        self.partition = self.kvstore_client != None

        # if not partition, not need to use kvstore_client
        if not self.partition:
            if shared_memory:
                local_rank = utils.local_rank()
            else:
                local_rank = 0

            if not shared_memory:
                self.node_memory = torch.zeros(
                    (num_nodes, dim_memory), dtype=torch.float32, device=device)
                self.node_memory_ts = torch.zeros(
                    num_nodes, dtype=torch.float32, device=device)
                self.mailbox = torch.zeros(
                    (num_nodes, self.dim_raw_message),
                    dtype=torch.float32, device=device)
                self.mailbox_ts = torch.zeros(
                    (num_nodes,), dtype=torch.float32, device=device)
            else:
                if local_rank == 0:
                    self.node_memory = create_shared_mem_array(
                        'node_memory', (num_nodes, dim_memory), dtype=torch.float32)
                    self.node_memory_ts = create_shared_mem_array(
                        'node_memory_ts', (num_nodes,), dtype=torch.float32)
                    self.mailbox = create_shared_mem_array(
                        'mailbox', (num_nodes, self.dim_raw_message),
                        dtype=torch.float32)
                    self.mailbox_ts = create_shared_mem_array(
                        'mailbox_ts', (num_nodes,), dtype=torch.float32)

                    self.node_memory.zero_()
                    self.node_memory_ts.zero_()
                    self.mailbox.zero_()
                    self.mailbox_ts.zero_()

                torch.distributed.barrier()

                if local_rank != 0:
                    # NB: `num_nodes` should be same for all local processes because
                    # they share the same local graph
                    self.node_memory = get_shared_mem_array(
                        'node_memory', (num_nodes, dim_memory), torch.float32)
                    self.node_memory_ts = get_shared_mem_array(
                        'node_memory_ts', (num_nodes,), torch.float32)
                    self.mailbox = get_shared_mem_array(
                        'mailbox', (num_nodes, self.dim_raw_message), torch.float32)
                    self.mailbox_ts = get_shared_mem_array(
                        'mailbox_ts', (num_nodes,), torch.float32)

    def reset(self):
        """
        Reset the memory and the mailbox.
        """
        if self.partition:
            self.kvstore_client.reset_memory()
        else:
            self.node_memory.fill_(0)
            self.node_memory_ts.fill_(0)
            self.mailbox.fill_(0)
            self.mailbox_ts.fill_(0)

    def resize(self, num_nodes):
        """
        Resize the memory and the mailbox.

        Args:
            num_nodes: number of nodes in the graph
        """
        if num_nodes <= self.num_nodes:
            return

        self.node_memory.resize_(num_nodes, self.dim_memory)
        self.node_memory_ts.resize_(num_nodes)
        self.mailbox.resize_(num_nodes, self.dim_raw_message)
        self.mailbox_ts.resize_(num_nodes,)

        # fill zeros for the new nodes
        self.node_memory[self.num_nodes:].fill_(0)
        self.node_memory_ts[self.num_nodes:].fill_(0)
        self.mailbox[self.num_nodes:].fill_(0)
        self.mailbox_ts[self.num_nodes:].fill_(0)

        self.num_nodes = num_nodes

    def backup(self) -> Dict:
        """
        Backup the current memory and mailbox.
        """
        return {
            'node_memory': self.node_memory.clone(),
            'node_memory_ts': self.node_memory_ts.clone(),
            'mailbox': self.mailbox.clone(),
            'mailbox_ts': self.mailbox_ts.clone(),
        }

    def restore(self, backup: Dict):
        """
        Restore the memory and mailbox from the backup.

        Args:
            backup: backup of the memory and mailbox
        """
        self.node_memory.copy_(backup['node_memory'])
        self.node_memory_ts.copy_(backup['node_memory_ts'])
        self.mailbox.copy_(backup['mailbox'])
        self.mailbox_ts.copy_(backup['mailbox_ts'])
    
    def findOverlapMem(self, data_loader, length: int, rank: int, world_size: int):
        iteration_now = -1
        last_nodes, current_nodes = None, None
        self.recv_msg, self.pull_msg, self.send_msg, self.push_msg = [], [], [], []
        data_list = []
        for all_nodes, _, _ in data_loader:
            all_nodes = all_nodes[:length]
            target_nodes = np.unique(all_nodes)
            data_list.append(target_nodes)
        for i in range(rank, len(data_list), world_size):
            if i-1 >= 0:
                last_nodes = data_list[i-1]
            else:
                last_nodes = None
            current_nodes = data_list[i]
            if i+1 < len(data_list):
                next_nodes = data_list[i+1]
            else:
                next_nodes = None
            
            if last_nodes is not None:
                self.recv_msg.append(torch.from_numpy(np.intersect1d(last_nodes, current_nodes)))
                self.pull_msg.append(torch.from_numpy(np.setdiff1d(current_nodes, self.recv_msg[-1])))
            else:
                self.recv_msg.append(None)
                self.pull_msg.append(torch.from_numpy(current_nodes))
            if next_nodes is not None:
                self.send_msg.append(torch.from_numpy(np.intersect1d(current_nodes, next_nodes)))
                self.push_msg.append(torch.from_numpy(np.setdiff1d(current_nodes, self.send_msg[-1])))
            else:
                self.send_msg.append(None)
                self.push_msg.append(torch.from_numpy(current_nodes))

        # print(f'{rank}recv: ', [msg[:5] for msg in self.recv_msg[:5]])
        # print(f'{rank}send: ', [msg[:5] for msg in self.send_msg[:5]])
    
    def recv_mem(self, iteration_now, rank, world_size, device, group = None):
        # Returns the memory required for the current iteratio, the memory required for send to next iteration
        cached_idx = self.recv_msg[iteration_now//world_size]
        if cached_idx is None:
            pass
        elif len(cached_idx) > 0:
            cached_mem = torch.empty([len(cached_idx), self.dim_memory], device=device)
            cached_mail = torch.empty([len(cached_idx), self.dim_raw_message], device=device)
            src = (rank-1+world_size)% world_size
            t1 = time.time()
            reqs = recv_req([cached_mem, cached_mail], rank, src, group)
            # print(time.time()-t1)
            # print('debug2 ', cached_mem)
        else:
            src = (rank-1+world_size)% world_size
            recv(None, rank, src, group)

        uncached_idx = self.pull_msg[iteration_now//world_size]
        uncached_mem = self.node_memory[uncached_idx].to(device)
        uncached_mail = self.mailbox[uncached_idx].to(device)
        if cached_idx is not None and len(cached_idx) > 0:
            for req in reqs:
                req.wait()
            idx = torch.cat((cached_idx, uncached_idx))
            mem = torch.cat((cached_mem, uncached_mem), dim=0)
            mail = torch.cat((cached_mail, uncached_mail), dim=0)
        else:
            idx = uncached_idx
            mem = uncached_mem
            mail = uncached_mail

        idx_idx = np.argsort(idx)
        mem = mem[idx_idx]
        mail = mail[idx_idx]
        return mem, mail

    def send_mem(self, mem, mail, rank, world_size, group):
        if mem is None:
            send_thread = None
        elif mem.shape[0] > 0:
            dst = (rank+1) % world_size
            send_thread = threading.Thread(target=send, args=([mem, mail], rank, dst, group))
            send_thread.start()
        else:
            dst = (rank+1) % world_size
            send_thread = threading.Thread(target=send, args=(None, rank, dst, group))
            send_thread.start()
        return send_thread
        
    def push_back_mem(self, mem, mail):
        pass