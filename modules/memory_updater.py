import torch
from dgl.heterograph import DGLBlock

from gnnflow.models.modules.layers import TimeEncode


class GRUMemeoryUpdater(torch.nn.Module):
    """
    GRU memory updater proposed by TGN
    """

    def __init__(self, dim_node: int, dim_edge: int, dim_time: int,
                 dim_embed: int, dim_memory: int):
        """
        Args:
            dim_node: dimension of node features/embeddings
            dim_edge: dimension of edge features
            dim_time: dimension of time features
            dim_embed: dimension of output embeddings
            dim_memory: dimension of memory
        """
        super(GRUMemeoryUpdater, self).__init__()
        self.dim_message = 2 * dim_memory + dim_edge
        self.dim_node = dim_node
        self.dim_time = dim_time
        self.dim_embed = dim_embed
        self.updater = torch.nn.GRUCell(
            self.dim_message + self.dim_time, dim_memory)

        self.use_time_enc = dim_time > 0
        if self.use_time_enc:
            self.time_enc = TimeEncode(dim_time)

        if dim_node > 0 and dim_node != dim_memory:
            self.node_feat_proj = torch.nn.Linear(dim_node, dim_memory)

    def forward(self, mem: torch.tensor, mail: torch.tensor, mem_ts: torch.tensor, mail_ts: torch.tensor):
        if self.use_time_enc:
            time_feat = self.time_enc(mail_ts-mem_ts)
            mail = torch.cat([mail, time_feat], dim=1)
        return self.updater(mail, mem)