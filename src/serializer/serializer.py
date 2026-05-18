from abc import ABC, abstractmethod
from typing import Any

from ..core.base import Function

class Serializer(Function, ABC):
    @abstractmethod
    def __call__(self, *, obj: Any) -> str:
        ...