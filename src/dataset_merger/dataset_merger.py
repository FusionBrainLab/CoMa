from abc import ABC, abstractmethod
from typing import List, Any, Dict

from ..core.base import Function

class DatasetMerger(Function, ABC):
    @abstractmethod
    def __call__(self, *, datasets: Dict[str, Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
        ...