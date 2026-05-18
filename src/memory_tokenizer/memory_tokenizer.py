from abc import ABC, abstractmethod
from typing import List, TypedDict, Dict, Any

import torch

from ..core.base import Function
from ..agent_message import AgentMessage

class MemoryTokenizer(Function, ABC):
    @abstractmethod
    def __call__(self, *, memories: List[List[AgentMessage]]) -> Dict[str, Any]:
        ...