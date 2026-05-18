from abc import ABC, abstractmethod
from typing import Dict, Any

from torch import nn

from ..core.base import Function

class ModelCreator(Function, ABC):
    @abstractmethod
    def __call__(self) -> nn.Module:
        ...