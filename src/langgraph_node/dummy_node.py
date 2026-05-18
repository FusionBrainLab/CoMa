from typing import List, Protocol, Dict, Any

from .langgraph_node import LanggraphNode

class DummyNode(LanggraphNode):
    def __call__(self, *, state: Dict[str, Any]) -> Dict[str, Any]:
        return state