from abc import ABC, abstractmethod
from typing import Any

from ..core.base import Function

class Logger(Function, ABC):
    @abstractmethod
    def __call__(self, **kwargs: Any) -> None:
        ...