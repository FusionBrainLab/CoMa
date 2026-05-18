from typing import Protocol, Dict, List, Generator, Any
from abc import ABC, abstractmethod
import os
import math

import torch

from .tensor_sharder import TensorSharder
from ..index_sharder import IndexSharder

class RangeTensorSharder(TensorSharder):
    def __init__(self, *, dim: int,
                        n_ranges: int,
                        index_sharder: IndexSharder) -> None:
        self.dim = dim
        self.n_ranges = n_ranges
        self.index_sharder = index_sharder

    def __call__(self, *, tensor: torch.Tensor, n_shards: int) -> List[torch.Tensor]:
        shard_indexes = self.index_sharder(n_indexes=self.n_ranges, n_shards=n_shards)
        dim_size = tensor.shape[self.dim]
        range_size = int(math.floor(dim_size / self.n_ranges))
        tensor_shards = []
        for i in range(n_shards):
            join_indexes = []
            for ind in shard_indexes[i]:
                local_indexes = list(range(range_size*ind, range_size*(ind+1)))
                join_indexes.extend(local_indexes)
            indices = torch.tensor(join_indexes)
            shard = torch.index_select(tensor, self.dim, indices)
            tensor_shards.append(shard)
        return tensor_shards