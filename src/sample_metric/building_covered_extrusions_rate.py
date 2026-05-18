from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from shapely import unary_union
from shapely.geometry import MultiPolygon
import trimesh

from .sample_metric import SampleMetric
from ..mesh_compiler import MassingMeshCompiler

class BuildingCoveredExtrusionsRate(SampleMetric):
    def __init__(self, *, building_key: str) -> None:
        self.building_key = building_key
        self.mesh_compiler = MassingMeshCompiler()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        building = sample[self.building_key]
        
        extrusion_meshes = []
        for extrusion in building["massing"]:
            mesh = self.mesh_compiler(obj=[{"id":"0", "massing":[extrusion]}])
            extrusion_meshes.append(mesh)

        covered = []
        for i in range(len(extrusion_meshes)):
            for j in range(i + 1, len(extrusion_meshes)):
                e1 = extrusion_meshes[i]
                e2 = extrusion_meshes[j]
                if i in covered or j in covered:
                    continue

                intersection = trimesh.boolean.intersection([e1, e2])
                if intersection.volume == e1.volume:
                    covered.append(i) 
                elif intersection.volume == e2.volume:
                    covered.append(j) 
                elif intersection.volume == e1.volume and intersection.volume == e2.volume:
                    covered.append(i)
        value = len(covered)/len(extrusion_meshes)
        return value