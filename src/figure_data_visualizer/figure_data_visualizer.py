from abc import ABC, abstractmethod
from typing import Any, Dict

from matplotlib.figure import Figure

from ..core.base import Function

class FigureDataVisualizer(Function, ABC):
    @abstractmethod
    def __call__(self, *, data: Dict[str, Any]) -> Figure:
        ...
