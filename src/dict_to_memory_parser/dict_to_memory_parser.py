from abc import ABC, abstractmethod
from typing import List, Any, Dict

from ..core.base import Function
from ..agent_message import AgentMessage, AgentMessageContent

class DictToMemoryParser(Function, ABC):
    @abstractmethod
    def __call__(self, *, dictionary: Dict[str, Any]) -> List[AgentMessage]:
        ...
