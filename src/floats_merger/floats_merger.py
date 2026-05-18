from abc import ABC, abstractmethod
from typing import List, Any, Dict

from ..core.base import Function

class FloatsMerger(Function, ABC):
    @abstractmethod
    def __call__(self, *, floats: List[float]) -> float:
        ...