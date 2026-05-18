from typing import Protocol, Dict, List, Generator, Any
from abc import ABC, abstractmethod
import os
import math

import torch

from .tensor_sharder import TensorSharder
from ..index_sharder import IndexSharder

class SingleTensorSharder(TensorSharder):
    def __init__(self, *, dim: int,
                        index_sharder: IndexSharder) -> None:
        self.dim = dim
        self.index_sharder = index_sharder

    def __call__(self, *, tensor: torch.Tensor, n_shards: int) -> List[torch.Tensor]:
        shard_indexes = self.index_sharder(n_indexes=tensor.shape[self.dim], n_shards=n_shards)
        tensor_shards = []
        for i in range(n_shards):
            indices = torch.tensor(shard_indexes[i])
            shard = torch.index_select(tensor, self.dim, indices)
            tensor_shards.append(shard)
        return tensor_shards