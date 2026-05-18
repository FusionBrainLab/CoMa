from typing import Protocol, Dict, List, Generator, Any
from abc import ABC, abstractmethod
import os

import torch

from ..core.base import Function

class TensorBatchProcessor(Function, ABC):
    @abstractmethod
    def __call__(self, *, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ...