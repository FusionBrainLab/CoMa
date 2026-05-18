from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Protocol, Dict, TypedDict, Any
from functools import partial

from langgraph.graph import StateGraph, START, END

from .core.base import Function
from .langgraph_node import LanggraphNode

@dataclass
class LanggraphEdgeConfig:
    key: str
    args: Dict[str, Any]

class LanggraphModule(Function):
    def __init__(self, *, nodes: Dict[str, LanggraphNode],
                        edges: List[LanggraphEdgeConfig]) -> None:
        self.nodes = nodes
        self.edges = edges

    def __call__(self, *, state: Dict[str, Any]) -> Dict[str, Any]:
        graph = StateGraph(Dict)
        for key, node in self.nodes.items():
            def wrapper(node, state):
                return node(state=state)
            graph.add_node(key, partial(wrapper, node))
        value_mapping = {
            "START":START,
            "END":END
        }
        for config in self.edges:
            args = config.args
            for k, v in args.items():
                if type(v) == type("string"):
                    args[k] = value_mapping.get(v, args[k])
            getattr(graph, config.key)(**args)
        graph = graph.compile()

        output = graph.invoke(state)
        return output
