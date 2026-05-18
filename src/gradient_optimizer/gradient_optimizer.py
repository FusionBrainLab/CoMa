from abc import ABC, abstractmethod
from typing import Dict, Any, Union, TypedDict

import torch
from torch import nn

from ..core.base import Function

class GradientOptimizer(Function, ABC):
    @abstractmethod
    def __call__(self, *, model: nn.Module) -> None:
        ...