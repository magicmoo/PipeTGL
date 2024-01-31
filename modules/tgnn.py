from typing import Dict, List, Optional, Union

import torch
from dgl.heterograph import DGLBlock
from gnnflow.distributed.kvstore import KVStoreClient
from gnnflow.models.modules.layers import EdgePredictor, TransfomerAttentionLayer
import sys
import os
from modules.memory import Memory
from modules.memory_updater import GRUMemeoryUpdater


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
    
    def update_memory_and_mail(self, b: DGLBlock, length: int, edge_feats: Optional[torch.Tensor]=None):

        device = b.device
        all_nodes = b.srcdata['ID'][:length]
        all_nodes_unique, inv = torch.unique(
            all_nodes.cpu(), return_inverse=True)
        
        mem = self.memory.node_memory[all_nodes_unique].to(device)
        mem_ts = self.memory.node_memory_ts[all_nodes_unique].to(device)
        mail = self.memory.mailbox[all_nodes_unique].to(device)
        mail_ts = self.memory.mailbox_ts[all_nodes_unique].to(device)

        mem = mem[inv]
        mem_ts = mem_ts[inv]
        mail = mail[inv]
        mail_ts = mail_ts[inv]

        new_memory = self.memory_updater.forward(mem, mail, mem_ts, mail_ts)

        last_updated_nid = all_nodes.to(self.memory.device)
        last_updated_memory = new_memory.to(self.memory.device)
        last_updated_ts = mail_ts.to(self.memory.device)

        # genereate mail
        split_chunks = 2
        if edge_feats is None:
            # dummy edge features
            edge_feats = torch.zeros(
                last_updated_nid.shape[0] // split_chunks, self.dim_edge,
                device=self.memory.device)


        edge_feats = edge_feats.to(self.memory.device)

        src, dst, *_ = last_updated_nid.tensor_split(split_chunks)
        mem_src, mem_dst, *_ = last_updated_memory.tensor_split(split_chunks)
        src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
        dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
        mail = torch.cat([src_mail, dst_mail],
                         dim=1).reshape(-1, src_mail.shape[1])
        nid = torch.cat(
            [src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
        mail_ts = last_updated_ts[:len(nid)]

        # find unique nid to update mailbox
        uni, inv = torch.unique(nid, return_inverse=True)
        perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
        nid = nid[perm]
        mail = mail[perm]
        mail_ts = mail_ts[perm]

        # prepare mem
        num_true_src_dst = last_updated_nid.shape[0] // split_chunks * 2
        nid = last_updated_nid[:num_true_src_dst].to(self.memory.device)
        memory = last_updated_memory[:num_true_src_dst].to(self.memory.device)
        ts = last_updated_ts[:num_true_src_dst].to(self.memory.device)
        # the nid of mem and mail is different
        # after unique they are the same but perm is still different
        uni, inv = torch.unique(nid, return_inverse=True)
        perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
        nid = nid[perm]
        mem = memory[perm]
        mem_ts = ts[perm]

        if self.memory.partition:
            # cat the memory first
            all_mem = torch.cat((mem,
                                 mem_ts.unsqueeze(dim=1),
                                 mail,
                                 mail_ts.unsqueeze(dim=1)),
                                dim=1)
            self.kvstore_client.push(nid, all_mem, mode='memory')
        else:
            # update mailbox first
            self.memory.mailbox[nid] = mail
            self.memory.mailbox_ts[nid] = mail_ts
            # update mem
            self.memory.node_memory[nid] = mem
            self.memory.node_memory_ts[nid] = mem_ts

    # def update_memory_and_mail(self, b: DGLBlock, length: int, edge_feats: Optional[torch.Tensor]=None):
    #     device = b.device
    #     all_nodes = b.srcdata['ID'][:length]
    #     mem = b.srcdata['mem'][:length]
    #     mem_ts = b.srcdata['mem_ts'][:length]
    #     mail = b.srcdata['mail'][:length]
    #     mail_ts = b.srcdata['mail_ts'][:length]

    #     new_memory = self.memory_updater(mem, mail, mem_ts, mail_ts)

    #     split_chunks = 2
    #     if edge_feats is None:
    #         # dummy edge features
    #         edge_feats = torch.zeros(
    #             mem.shape[0] // split_chunks, self.dim_edge,
    #             device=device)
    #     edge_feats = edge_feats.to(device)

    #     mem_src, mem_dst = new_memory.tensor_split(split_chunks)
    #     src, dst = all_nodes.tensor_split(split_chunks)
    #     src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
    #     dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
    #     new_mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1])
    #     new_mem = torch.cat([mem_src, mem_dst], dim=1).reshape(-1, mem_src.shape[1])
    #     nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
        
    #     ts_src, ts_dst = mail_ts.tensor_split(split_chunks)
    #     new_mem_ts = torch.cat([ts_src.unsqueeze(1), ts_dst.unsqueeze(1)], dim=1).reshape(-1)
    #     ts_src, ts_dst = b.srcdata['ts'][:length].tensor_split(split_chunks)
    #     new_mail_ts = torch.cat([ts_src.unsqueeze(1), ts_dst.unsqueeze(1)], dim=1).reshape(-1)
        
    #     # aggregate the mail
    #     uni, inv = torch.unique(nid, return_inverse=True)
    #     perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    #     perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
    #     nid = nid[perm]
    #     new_mail = new_mail[perm]
    #     new_mail_ts = new_mail_ts[perm]
    #     new_mem_ts = new_mem_ts[perm]
    #     new_mem = new_mem[perm]

    #     if self.memory.partition:
    #         # to be continue
    #         pass
    #     else:
    #         self.memory.node_memory[nid] = new_mem
    #         self.memory.node_memory_ts[nid] = new_mem_ts
    #         self.memory.mailbox[nid] = new_mail
    #         self.memory.mailbox_ts[nid] = new_mail_ts
    #     return new_memory
    
    def get_updated_memory(self, b: DGLBlock, updated_memory: torch.tensor, length: int):
        memory = self.memory_updater(b.srcdata['mem'], b.srcdata['mail'],
                                         b.srcdata['mem_ts'], b.srcdata['mail_ts'])
        if 'h' in b.srcdata:
            b.srcdata['h'] += memory
        else:
            b.srcdata['h'] = memory