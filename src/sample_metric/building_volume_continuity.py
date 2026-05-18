from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric
from ..mesh_compiler import MassingMeshCompiler

class BuildingVolumeContinuity(SampleMetric):
    def __init__(self, *, building_key: str) -> None:
        self.building_key = building_key
        self.mesh_compiler = MassingMeshCompiler()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        building = sample[self.building_key]
        mesh = self.mesh_compiler(obj=[building])
        parts = mesh.split()
        value = 1/len(parts)
        return value