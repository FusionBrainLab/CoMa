from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Protocol, Dict, Any

from ..core.base import Function

class LanggraphNode(Function, ABC):
    @abstractmethod
    def __call__(self, *, state: Dict[str, Any]) -> Dict[str, Any]:
        ...