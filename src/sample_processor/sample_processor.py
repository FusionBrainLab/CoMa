from abc import ABC, abstractmethod
from typing import Any, Dict

from ..core.base import Function

class SampleProcessor(Function, ABC):
    @abstractmethod
    def __call__(self, *, sample: Dict[str, Any]) -> Dict[str, Any]:
        ...