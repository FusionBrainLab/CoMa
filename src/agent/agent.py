from abc import ABC, abstractmethod
from typing import List

from ..core.base import Function
from ..agent_message import AgentMessage, AgentMessageContent
    
class Agent(Function, ABC):
    @abstractmethod
    def __call__(self, *, memories: List[List[AgentMessage]]) -> List[List[AgentMessage]]:
        ...