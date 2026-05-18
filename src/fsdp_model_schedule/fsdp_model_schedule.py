from abc import ABC, abstractmethod

from ..core.base import Function

class FSDPModelSchedule(Function, ABC):
    @abstractmethod
    def __call__(self, *, forward: int) -> bool:
        ...