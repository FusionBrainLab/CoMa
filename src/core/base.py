from typing import Any
from abc import ABC, abstractmethod

from .meta import Meta

class Function(ABC, metaclass=Meta):
    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        ...

class Container(ABC, metaclass=Meta):
    pass