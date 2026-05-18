from typing import Any, List, Dict
from abc import ABC, abstractmethod

from ..core.base import Container

class Storage(Container, ABC):
    @abstractmethod
    def set_path(self, *, path: str) -> None:
        ...
    
    @abstractmethod
    def load(self) -> object:
        ...
    
    @abstractmethod
    def save(self, *, obj: object) -> None:
        ...