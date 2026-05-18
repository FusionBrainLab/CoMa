from abc import ABC, abstractmethod
from typing import Any, Dict

from ..core.base import Function

class SaveVisualizer(Function, ABC):
    @abstractmethod
    def __call__(self, *, data: Dict[str, Any]) -> str:
        ...