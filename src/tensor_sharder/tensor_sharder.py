from typing import Protocol, Dict, List, Generator, Any
from abc import ABC, abstractmethod
import os

import torch

from ..core.base import Function

class TensorSharder(Function, ABC):
    @abstractmethod
    def __call__(self, *, tensor: torch.Tensor, n_shards: int) -> List[torch.Tensor]:
        ...