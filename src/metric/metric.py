from abc import ABC, abstractmethod
from typing import List, Any, Dict

from ..core.base import Function

class Metric(Function, ABC):
    @abstractmethod
    def __call__(self, *, submit: Dict[str, List[Any]]) -> float:
        ...