from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Protocol
import re

from ..core.base import Function
from ..agent_message import AgentMessage, AgentMessageContent
from .memory_indicator import MemoryIndicator

class ContentPatternMemoryIndicator(MemoryIndicator):
    def __init__(self, *, message_ind: int,
                        pattern: str) -> None:
        self.message_ind = message_ind
        self.pattern = pattern

    def __call__(self, *, memory: List[AgentMessage]) -> bool:
        message = memory[self.message_ind]
        content = "".join([c.content for c in message.content])
        search_result = re.findall(self.pattern, content)
        return len(search_result) > 0