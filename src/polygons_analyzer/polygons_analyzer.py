from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple

from ..core.base import Function

class PolygonsAnalyzer(Function, ABC):
    @abstractmethod
    def __call__(self, *, polygons: List[List[Tuple[float, float]]]) -> float:
        ...