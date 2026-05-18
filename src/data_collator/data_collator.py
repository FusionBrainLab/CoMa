from typing import Dict, List, Any
from abc import ABC, abstractmethod
import os

import torch

from ..core.base import Function

class DataCollator(Function, ABC):
    @abstractmethod
    def __call__(self, *, batch: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        ...