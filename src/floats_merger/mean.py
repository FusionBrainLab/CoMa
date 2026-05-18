from typing import Any, Dict, List

from .floats_merger import FloatsMerger

class Mean(FloatsMerger):
    def __call__(self, *, floats: List[float]) -> float:
        output = sum(floats)/len(floats)
        return output