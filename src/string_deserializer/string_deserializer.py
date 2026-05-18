from abc import ABC, abstractmethod
from typing import Any

from ..core.base import Function

class StringDeserializer(Function, ABC):
    @abstractmethod
    def __call__(self, *, string: str) -> Any:
        ...