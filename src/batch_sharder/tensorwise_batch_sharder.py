from typing import Protocol, Dict, List, Generator, Any
from abc import ABC, abstractmethod
import os
import math

import torch

from .batch_sharder import BatchSharder
from ..tensor_sharder import TensorSharder

class TensorwiseBatchSharder(BatchSharder):
    def __init__(self, *, key_sharders: Dict[str, TensorSharder]) -> None:
        self.key_sharders = key_sharders

    def __call__(self, *, batch: Dict[str, torch.Tensor], n_shards: int) -> List[Dict[str, torch.Tensor]]:
        batch_shards = [{} for _ in range(n_shards)]
        for key in batch.keys():
            tensor_shards = self.key_sharders[key](tensor=batch[key], n_shards=n_shards)
            for i in range(n_shards):
                batch_shards[i][key] = tensor_shards[i]
        return batch_shards