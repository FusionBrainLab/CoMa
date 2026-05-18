from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

import trimesh

from .sample_metric import SampleMetric
from ..mesh_compiler import MassingMeshCompiler

class MassingBuildingsIntersectionRate(SampleMetric):
    def __init__(self, *, massing_key: str) -> None:
        self.massing_key = massing_key
        self.mesh_compiler = MassingMeshCompiler()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        massing = sample[self.massing_key]
        meshes = [self.mesh_compiler(obj=[m]) for m in massing]
        
        values = []
        for i in range(len(meshes)):
            for j in range(i + 1, len(meshes)):
                b1 = meshes[i]
                b2 = meshes[j]
                intersection = trimesh.boolean.intersection([b1, b2])
                union = trimesh.boolean.union([b1, b2])
                iou = intersection.volume/union.volume
                values.append(iou)
        value = sum(values)/len(values)
        return value