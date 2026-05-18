from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from ..core.base import Function

class GradientNormComputer(Function, ABC):
    @abstractmethod
    def __call__(self, *, model: nn.Module) -> float:
        ...