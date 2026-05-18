from abc import ABC, abstractmethod
from typing import List, Any, Dict

from ..core.base import Function

class DatasetHandler(Function, ABC):
    @abstractmethod
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> None:
        ...