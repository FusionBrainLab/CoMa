from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Protocol
import re

from ..core.base import Function
from ..agent_message import AgentMessage, AgentMessageContent

class MemoryIndicator(Function, ABC):
    @abstractmethod
    def __call__(self, *, memory: List[AgentMessage]) -> bool:
        ...
