from abc import ABC, abstractmethod
from typing import Optional, TypedDict, Dict

from torch import nn
import json

from ..core.base import Function

class TrainingStrategy(Function, ABC):
    @abstractmethod
    def __call__(self, *, model: nn.Module) -> nn.Module:
        ...