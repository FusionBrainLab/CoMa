from abc import ABC, abstractmethod
from typing import List, Dict, Any
import re

from ..core.base import Function
import trimesh

class MeshCompiler(Function, ABC):
    @abstractmethod
    def __call__(self, *, obj: Any) -> trimesh.Trimesh:
        ...