from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
import json

from .core.base import Function

class MDPointConverter(Function):
    def __call__(self, *, md_point: str) -> Tuple[float, float]:
        x, y = md_point.split(",")
        point = (float(y), float(x))
        return point