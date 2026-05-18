from typing import List, Protocol, Dict, Any

from .langgraph_node import LanggraphNode
from ..core.base import Function

class FunctionNode(LanggraphNode):
    def __init__(self, *, function: Function) -> None:
        self.function = function

    def __call__(self, *, state: Dict[str, Any]) -> Dict[str, Any]:
        output = self.function(**state)
        state.update(output)
        return state