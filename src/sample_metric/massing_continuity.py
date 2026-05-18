from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

import trimesh
import shapely
from trimesh import transformations
import numpy as np

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter
from ..mesh_compiler import MassingMeshCompiler

class MassingContinuity(SampleMetric):
    def __init__(self, *, pred_col: str) -> None:
        self.pred_col = pred_col
        self.mesh_creator = MassingMeshCompiler()
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        total_pred_massings = sample[self.pred_col]

        value = 0
        computed = 0
        for pred in total_pred_massings:
            pred_mesh = self.mesh_creator(obj=[pred])
            components = pred_mesh.split(only_watertight=False)

            local_value = max(1 - (len(components) - 1) / len(pred["massing"]), 0)
            value += local_value
            computed += 1

        result = value / computed if computed > 0 else 0
        return result