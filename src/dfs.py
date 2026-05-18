from abc import ABC, abstractmethod
from typing import List, Any, Dict

from .core.base import Function

class DFS(Function):
    def __init__(self) -> None:
        def dfs(graph: Dict[str, List[str]], node_id: str, visited: List[str]):
            visited.append(node_id)
            for neighbor in graph[node_id]:
                if neighbor not in visited:
                    self.dfs(graph, neighbor, visited)
        self.dfs = dfs

    def __call__(self, *, graph: Dict[str, List[str]], start_node_id: str) -> List[str]:
        visited = []
        self.dfs(graph=graph, node_id=start_node_id, visited=visited)
        return visited

