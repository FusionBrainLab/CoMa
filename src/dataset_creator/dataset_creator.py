from abc import ABC, abstractmethod
from typing import List, Any, Dict

from ..core.base import Function

class DatasetCreator(Function, ABC):
    @abstractmethod
    def __call__(self) -> Dict[str, List[Any]]:
        ...
