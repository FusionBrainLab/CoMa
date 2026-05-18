from abc import ABC, abstractmethod

from torch import nn

from ..core.base import Function

class ModelHandler(Function, ABC):
    @abstractmethod
    def __call__(self, *, model: nn.Module) -> None:
        ...