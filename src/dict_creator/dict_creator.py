from abc import ABC, abstractmethod
from typing import List, Any, Dict

from ..core.base import Function

class DictCreator(Function, ABC):
    @abstractmethod
    def __call__(self) -> Dict[str, Any]:
        ...