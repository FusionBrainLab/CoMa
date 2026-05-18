from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
from torch import nn

from ..core.base import Function

class GradientComputer(Function, ABC):
    @abstractmethod
    def __call__(self, *, model: nn.Module, batch: Dict[str, torch.Tensor]) -> None:
        ...