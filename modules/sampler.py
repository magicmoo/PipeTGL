import collections
import re
from typing import Iterable, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed
from torch._six import string_classes
from torch.utils.data import BatchSampler, Dataset, Sampler

from gnnflow.utils import DstRandEdgeSampler, RandEdgeSampler, local_rank

class DistributedBatchSampler(BatchSampler):
    """
    Distributed batch sampler.
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]],
                 batch_size: int, drop_last: bool,
                 rank: int, world_size: int,
                 num_chunks: int = 1):
        """
        Args:
            sampler: Base class for all Samplers.
            batch_size: Size of mini-batch.
            drop_last: Set to ``True`` to drop the last incomplete batch, if the
                dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the
                last batch will be smaller.
            rank: The rank of the current process.
            world_size: The number of processes.
            num_chunks: Number of chunks to split the batch into.
        """
        super(DistributedBatchSampler, self).__init__(sampler, batch_size,
                                                      drop_last)
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank()
        self.device = torch.device('cuda', self.local_rank)
        assert 0 < num_chunks < batch_size, "num_chunks must be in (0, batch_size)"

        self.num_chunks = num_chunks
        self.chunk_size = batch_size // num_chunks
        self.reorder = False
        self.random_size = batch_size

    def __iter__(self) -> Iterator[List[int]]:
        self.reset()
        batch = []
        for idx in self.sampler:
            # if idx % self.world_size != self.rank:
            if (idx // self.batch_size)%self.world_size != self.rank:
                continue
            batch.append(idx)
            if self.reorder:
                if len(batch) == self.random_size:
                    yield batch
                    self.reorder = False
                    batch = []
            else:
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def reset(self):
        self.reorder = self.num_chunks > 1
        if self.reorder:
            if self.rank == 0:
                randint = torch.randint(
                    0, self.num_chunks, size=(1,), device=self.device)
            else:
                randint = torch.zeros(1, dtype=torch.int64, device=self.device)

            torch.distributed.broadcast(randint, src=0)
            self.random_size = int(randint.item() * self.chunk_size)
            if self.random_size == 0:
                self.reorder = False