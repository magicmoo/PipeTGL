from typing import Dict, List, Optional, Union

from networkx import all_node_cuts
import torch
from dgl.heterograph import DGLBlock
from gnnflow.distributed.kvstore import KVStoreClient
from gnnflow.models.modules.layers import EdgePredictor, TransfomerAttentionLayer
import sys
import os
import numpy as np
import threading
from modules.memory import Memory
from modules.memory_updater import GRUMemeoryUpdater
import time
import torch.distributed as dist


class TGNN(torch.nn.Module):

    def __init__(self, dim_node: int, dim_edge: int, dim_time: int,
                 dim_embed: int, num_layers: int, num_snapshots: int,
                 att_head: int, dropout: float, att_dropout: float,
                 use_memory: bool, dim_memory: Optional[int] = None,
                 num_nodes: Optional[int] = None,
                 memory_device: Union[torch.device, str] = 'cpu',
                 memory_shared: bool = False,
                 kvstore_client: Optional[KVStoreClient] = None,
                 *args, **kwargs):
        """
        Args:
            dim_node: dimension of node features/embeddings
            dim_edge: dimension of edge features
            dim_time: dimension of time features
            dim_embed: dimension of output embeddings
            num_layers: number of layers
            num_snapshots: number of snapshots
            att_head: number of heads for attention
            dropout: dropout rate
            att_dropout: dropout rate for attention
            use_memory: whether to use memory
            dim_memory: dimension of memory
            num_nodes: number of nodes in the graph
            memory_device: device of the memory
            memory_shared: whether to share memory across local workers
            kvstore_client: The KVStore_Client for fetching memorys when using partition
        """
        super(TGNN, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.dim_time = dim_time
        self.dim_embed = dim_embed
        self.num_layers = num_layers
        self.num_snapshots = num_snapshots
        self.att_head = att_head
        self.dropout = dropout
        self.att_dropout = att_dropout
        self.use_memory = use_memory

        if self.use_memory:
            assert num_snapshots == 1, 'memory is not supported for multiple snapshots'
            assert dim_memory is not None, 'dim_memory should be specified'
            assert num_nodes is not None, 'num_nodes is required when using memory'

            self.memory = Memory(num_nodes, dim_edge, dim_memory,
                                 memory_device, memory_shared,
                                 kvstore_client)

            self.memory_updater = GRUMemeoryUpdater(
                dim_node, dim_edge, dim_time, dim_embed, dim_memory)
            dim_node = dim_memory

        self.layers = torch.nn.ModuleDict()
        for l in range(num_layers):
            for h in range(num_snapshots):
                if l == 0:
                    dim_node_input = dim_node
                else:
                    dim_node_input = dim_embed

                key = 'l' + str(l) + 'h' + str(h)
                self.layers[key] = TransfomerAttentionLayer(dim_node_input,
                                                            dim_edge,
                                                            dim_time,
                                                            dim_embed,
                                                            att_head,
                                                            dropout,
                                                            att_dropout)

        if self.num_snapshots > 1:
            self.combiner = torch.nn.RNN(
                dim_embed, dim_embed)

        self.last_updated = None
        self.edge_predictor = EdgePredictor(dim_embed)

    def reset(self):
        if self.use_memory:
            self.memory.reset()

    def resize(self, num_nodes: int):
        if self.use_memory:
            self.memory.resize(num_nodes)

    def has_memory(self):
        return self.use_memory

    def backup_memory(self) -> Dict:
        if self.use_memory:
            return self.memory.backup()
        return {}

    def restore_memory(self, backup: Dict):
        if self.use_memory:
            self.memory.restore(backup)

    def forward(self, mfgs: List[List[DGLBlock]], return_embed: bool =False):
        """
        Args:
            mfgs: list of list of DGLBlocks
            neg_sample_ratio: negative sampling ratio
        """
        out = list()
        for l in range(self.num_layers):
            for h in range(self.num_snapshots):
                key = 'l' + str(l) + 'h' + str(h)
                rst = self.layers[key](mfgs[l][h])
                if l != self.num_layers - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)

        if self.num_snapshots == 1:
            embed = out[0]
        else:
            embed = torch.stack(out, dim=0)
            embed = self.combiner(embed)[0][-1, :, :]

        if return_embed:
            return embed
        return self.edge_predictor(embed)
    
    def update_memory_and_mail(self, b: DGLBlock, length: int, edge_feats: Optional[torch.Tensor] = None):
        t1 = time.time()
        device = b.device
        all_nodes = b.srcdata['ID'][:length]
        all_nodes_unique, inv = torch.unique(
            all_nodes.cpu(), return_inverse=True)
        # print(all_nodes_unique.shape, 'hello')
        
        mem = self.memory.node_memory[all_nodes_unique].to(device)
        mail = self.memory.mailbox[all_nodes_unique].to(device)
        mem_ts = self.memory.node_memory_ts[all_nodes_unique].to(device)
        mail_ts = self.memory.mailbox_ts[all_nodes_unique].to(device)

        t2 = time.time()

        updated_memory = self.memory_updater.forward(mem, mail, mem_ts, mail_ts)
        new_memory = updated_memory.clone().detach()
        new_memory = new_memory[inv]

        t3 = time.time()
        
        with torch.no_grad():
            last_updated_nid = all_nodes
            last_updated_memory = new_memory
            # last_updated_ts = mail_ts
            last_updated_ts = b.srcdata['ts'][:length]
            last_updated_mail_ts = mail_ts[inv]

            # genereate mail
            split_chunks = 2
            if edge_feats is None:
                # dummy edge features
                edge_feats = torch.zeros(
                    last_updated_nid.shape[0] // split_chunks, self.dim_edge,
                    device=self.memory.device)

            edge_feats = edge_feats.to(device)

            src, dst, *_ = last_updated_nid.tensor_split(split_chunks)
            mem_src, mem_dst, *_ = last_updated_memory.tensor_split(split_chunks)
            src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
            dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
            mail = torch.cat([src_mail, dst_mail],
                            dim=1).reshape(-1, src_mail.shape[1])
            nid = torch.cat(
                [src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
            # mail_ts = last_updated_ts[:len(nid)]
            src_mail_ts, dst_mail_ts, *_ = last_updated_ts.tensor_split(split_chunks)
            mail_ts = torch.cat(
                [src_mail_ts.unsqueeze(1), dst_mail_ts.unsqueeze(1)], dim=1).reshape(-1)

            # find unique nid to update mailbox
            uni, inv = torch.unique(nid, return_inverse=True)
            perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
            perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
            nid = nid[perm]
            mail = mail[perm]
            mail_ts = mail_ts[perm]

            # prepare mem
            num_true_src_dst = last_updated_nid.shape[0] // split_chunks * 2
            nid = last_updated_nid[:num_true_src_dst]
            memory = last_updated_memory[:num_true_src_dst]
            ts = last_updated_mail_ts[:num_true_src_dst]
            # the nid of mem and mail is different
            # after unique they are the same but perm is still different
            uni, inv = torch.unique(nid, return_inverse=True)
            perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
            perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
            nid = nid[perm]
            mem = memory[perm]
            mem_ts = ts[perm]

            t4 = time.time()

            if self.memory.partition:
                # cat the memory first
                pass
            else:
                self.memory.mailbox_ts[nid] = mail_ts.to(self.memory.device)
                self.memory.node_memory_ts[nid] = mem_ts.to(self.memory.device)
                self.memory.node_memory[nid] = mem.to(self.memory.device)
                self.memory.mailbox[nid] = mail.to(self.memory.device)
            t5 = time.time()
            # print("--------------")
            # print(t2-t1)
            # print(t3-t2)
            # print(t4-t3)
            # print(t5-t4)
            # print(time.time()-t1)
            return updated_memory, all_nodes_unique

    def update_memory(self, b: DGLBlock, length: int, mem: torch.tensor = None, mail: torch.tensor = None, 
                               push_msg: torch.tensor = None, send_msg: torch.tensor = None, edge_feats: Optional[torch.Tensor] = None):
        t1 = time.time()
        device = b.device
        all_nodes = b.srcdata['ID'][:length]
        all_nodes_unique, inv = torch.unique(
            all_nodes.cpu(), return_inverse=True)
        if mem is None:
            mem = self.memory.node_memory[all_nodes_unique].to(device)
        if mail is None:
            mail = self.memory.mailbox[all_nodes_unique].to(device)
        mem_ts = self.memory.node_memory_ts[all_nodes_unique].to(device)
        mail_ts = self.memory.mailbox_ts[all_nodes_unique].to(device)

        t2 = time.time()

        updated_memory = self.memory_updater(mem, mail, mem_ts, mail_ts)
        new_memory = updated_memory.clone().detach()
        new_memory = new_memory[inv]
        
        # mail_ts = mail_ts[inv]

        t3 = time.time()
        
        with torch.no_grad():
            last_updated_nid = all_nodes
            last_updated_memory = new_memory
            # last_updated_ts = mail_ts
            last_updated_ts = b.srcdata['ts'][:length]
            last_updated_mail_ts = mail_ts[inv]

            # genereate mail
            split_chunks = 2
            if edge_feats is None:
                # dummy edge features
                edge_feats = torch.zeros(
                    last_updated_nid.shape[0] // split_chunks, self.dim_edge,
                    device=self.memory.device)

            edge_feats = edge_feats.to(device)

            src, dst, *_ = last_updated_nid.tensor_split(split_chunks)
            mem_src, mem_dst, *_ = last_updated_memory.tensor_split(split_chunks)
            src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
            dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
            mail = torch.cat([src_mail, dst_mail],
                            dim=1).reshape(-1, src_mail.shape[1])
            nid = torch.cat(
                [src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
            # mail_ts = last_updated_ts
            src_mail_ts, dst_mail_ts, *_ = last_updated_ts.tensor_split(split_chunks)
            mail_ts = torch.cat(
                [src_mail_ts.unsqueeze(1), dst_mail_ts.unsqueeze(1)], dim=1).reshape(-1)

            # find unique nid to update mailbox
            uni, inv = torch.unique(nid, return_inverse=True)
            perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
            perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)

            nid = nid[perm]
            mail = mail[perm]
            mail_ts = mail_ts[perm]

            # prepare mem
            num_true_src_dst = last_updated_nid.shape[0] // split_chunks * 2
            nid = last_updated_nid[:num_true_src_dst]
            memory = last_updated_memory[:num_true_src_dst]
            ts = last_updated_mail_ts[:num_true_src_dst]
            # the nid of mem and mail is different
            # after unique they are the same but perm is still different
            uni, inv = torch.unique(nid, return_inverse=True)
            perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
            perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
            nid = nid[perm]
            mem = memory[perm]
            mem_ts = ts[perm]

            t4 = time.time()

            if self.memory.partition:
                # cat the memory first
                pass
            else:
                self.memory.mailbox_ts[nid] = mail_ts.to(self.memory.device)
                self.memory.node_memory_ts[nid] = mem_ts.to(self.memory.device)
                if send_msg is not None:
                    idx = torch.searchsorted(nid, push_msg.to(device))
                    self.memory.node_memory[nid[idx]] = mem[idx].to(self.memory.device)
                    self.memory.mailbox[nid[idx]] = mail[idx].to(self.memory.device)
                else:
                    self.memory.node_memory[nid] = mem.to(self.memory.device)
                    self.memory.mailbox[nid] = mail.to(self.memory.device)
            t5 = time.time()
            # print("--------------")
            # print(t2-t1)
            # print(t3-t2)
            # print(t4-t3)
            # print(t5-t4)

        if send_msg is not None:
            idx = torch.searchsorted(nid, send_msg.to(device))
            return updated_memory, all_nodes_unique, mem[idx], mail[idx]
        else:
            return updated_memory, all_nodes_unique, None, None
    
    def update_memory_and_send(self, b: DGLBlock, length: int, rank: int, world_size: int, group, mem: torch.tensor = None, mail: torch.tensor = None, 
                               push_msg: torch.tensor = None, send_msg: torch.tensor = None, edge_feats: Optional[torch.Tensor] = None, node_dst=-1):
        t1 = time.time()
        device = b.device
        all_nodes = b.srcdata['ID'][:length]
        all_nodes_unique, inv = torch.unique(
            all_nodes.cpu(), return_inverse=True)
        if mem is None:
            mem = self.memory.node_memory[all_nodes_unique].to(device)
        if mail is None:
            mail = self.memory.mailbox[all_nodes_unique].to(device)
        mem_ts = self.memory.node_memory_ts[all_nodes_unique].to(device)
        mail_ts = self.memory.mailbox_ts[all_nodes_unique].to(device)

        t2 = time.time()

        updated_memory = self.memory_updater(mem, mail, mem_ts, mail_ts)
        new_memory = updated_memory.clone().detach()
        new_memory = new_memory[inv]
        
        # mail_ts = mail_ts[inv]

        t3 = time.time()
        
        with torch.no_grad():
            last_updated_nid = all_nodes
            last_updated_memory = new_memory
            # last_updated_ts = mail_ts
            last_updated_ts = b.srcdata['ts'][:length]
            last_updated_mail_ts = mail_ts[inv]

            # genereate mail
            split_chunks = 2
            if edge_feats is None:
                # dummy edge features
                edge_feats = torch.zeros(
                    last_updated_nid.shape[0] // split_chunks, self.dim_edge,
                    device=self.memory.device)

            edge_feats = edge_feats.to(device)

            src, dst, *_ = last_updated_nid.tensor_split(split_chunks)
            mem_src, mem_dst, *_ = last_updated_memory.tensor_split(split_chunks)
            src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
            dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
            mail = torch.cat([src_mail, dst_mail],
                            dim=1).reshape(-1, src_mail.shape[1])
            nid = torch.cat(
                [src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
            # mail_ts = last_updated_ts
            src_mail_ts, dst_mail_ts, *_ = last_updated_ts.tensor_split(split_chunks)
            mail_ts = torch.cat(
                [src_mail_ts.unsqueeze(1), dst_mail_ts.unsqueeze(1)], dim=1).reshape(-1)

            # find unique nid to update mailbox
            uni, inv = torch.unique(nid, return_inverse=True)
            perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
            perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)

            nid = nid[perm]
            mail = mail[perm]
            mail_ts = mail_ts[perm]

            # prepare mem
            num_true_src_dst = last_updated_nid.shape[0] // split_chunks * 2
            nid = last_updated_nid[:num_true_src_dst]
            memory = last_updated_memory[:num_true_src_dst]
            ts = last_updated_mail_ts[:num_true_src_dst]
            # the nid of mem and mail is different
            # after unique they are the same but perm is still different
            uni, inv = torch.unique(nid, return_inverse=True)
            perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
            perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
            nid = nid[perm]
            mem = memory[perm]
            mem_ts = ts[perm]

            t4 = time.time()

            if self.memory.partition:
                # cat the memory first
                pass
            else:
                self.memory.mailbox_ts[nid] = mail_ts.to(self.memory.device)
                self.memory.node_memory_ts[nid] = mem_ts.to(self.memory.device)
                if send_msg is not None:
                    idx = torch.searchsorted(nid, push_msg.to(device))
                    self.memory.node_memory[nid[idx]] = mem[idx].to(self.memory.device)
                    self.memory.mailbox[nid[idx]] = mail[idx].to(self.memory.device)
                else:
                    self.memory.node_memory[nid] = mem.to(self.memory.device)
                    self.memory.mailbox[nid] = mail.to(self.memory.device)
            t5 = time.time()
            # print("--------------")
            # print(t2-t1)
            # print(t3-t2)
            # print(t4-t3)
            # print(t5-t4)

        if send_msg is not None:
            idx = torch.searchsorted(nid, send_msg.to(device))
            # print(send_msg.shape[0] / nid.shape[0])
            sends_thread1 = self.memory.send_mem(mem[idx], mail[idx], rank, world_size, group, dst=node_dst)
            # self.memory.node_memory[nid[idx]] = mem[idx].to(self.memory.device)
            # self.memory.mailbox[nid[idx]] = mail[idx].to(self.memory.device)
            return updated_memory, all_nodes_unique, sends_thread1
        else:
            return updated_memory, all_nodes_unique, None
        
    def prepare_input(self, b: DGLBlock, updated_memory: torch.tensor, overlap_nid: torch.tensor, input=None):
        device = b.device
        all_nodes = b.srcdata['ID']

        all_nodes_unique, _ = torch.unique(
            all_nodes.cpu(), return_inverse=True)
        
        if input is None:
            pull_nodes = torch.from_numpy(np.setdiff1d(all_nodes_unique.numpy(), overlap_nid.numpy()))
            mem = self.memory.node_memory[pull_nodes].to(device)
            mem_ts = self.memory.node_memory_ts[pull_nodes].to(device)
            mail = self.memory.mailbox[pull_nodes].to(device)
            mail_ts = self.memory.mailbox_ts[pull_nodes].to(device)
            # print(pull_nodes.shape)
        else:
            pull_nodes, mem, mem_ts, mail, mail_ts = input
        
        new_memory = self.memory_updater(mem, mail, mem_ts, mail_ts)
        memory = torch.cat((updated_memory, new_memory), dim=0)
        nid = torch.cat((overlap_nid, pull_nodes), dim=0).to(device)
        sorted_res = torch.sort(nid)
        nid = sorted_res.values
        memory = memory[sorted_res.indices]

        inv = torch.searchsorted(nid, all_nodes)
        if 'h' in b.srcdata:
            b.srcdata['h'] = memory[inv] +  self.memory_updater.node_feat_proj(b.srcdata['h'])
        else:
            b.srcdata['h'] = memory[inv]
        

    def get_updated_memory(self, b: DGLBlock, updated_memory: torch.tensor, mem, mail, mem_ts, mail_ts):
        memory = self.memory_updater(mem, mail, mem_ts, mail_ts)
        if 'h' in b.srcdata:
            b.srcdata['h'] += torch.cat((updated_memory, memory), dim=0)
        else:
            b.srcdata['h'] = torch.cat((updated_memory, memory), dim=0)