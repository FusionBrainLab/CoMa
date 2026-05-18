from abc import ABC, abstractmethod
from typing import List, Any, Dict

from ..core.base import Function

class SampleMetric(Function, ABC):
    @abstractmethod
    def __call__(self, *, sample: Dict[str, Any]) -> float:
        ...