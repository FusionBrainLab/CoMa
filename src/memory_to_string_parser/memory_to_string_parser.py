from abc import ABC, abstractmethod
from typing import List

from ..core.base import Function
from ..agent_message import AgentMessage

class MemoryToStringParser(Function, ABC):
    @abstractmethod
    def __call__(self, *, memory: List[AgentMessage]) -> str:
        ...
        