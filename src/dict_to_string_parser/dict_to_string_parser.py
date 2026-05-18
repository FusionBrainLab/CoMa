from abc import ABC, abstractmethod
from typing import List, Any, Dict

from ..core.base import Function

class DictToStringParser(Function, ABC):
    @abstractmethod
    def __call__(self, *, dictionary: Dict[str, Any]) -> str:
        ...