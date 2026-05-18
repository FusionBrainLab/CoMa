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

class ExtrusionsSelfIoU(SampleMetric):
    def __init__(self, *, pred_col: str) -> None:
        self.pred_col = pred_col
        self.mesh_creator = MassingMeshCompiler()
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        total_pred_massings = sample[self.pred_col]

        value = 0
        computed = 0
        for pred in total_pred_massings:
            for i, e in enumerate(pred["massing"]):
                base_mesh = self.mesh_creator(obj=[{"id":"0", "massing":[e]}])
                disjoint_mesh = self.mesh_creator(obj=[{"id":"0", "massing":[pred["massing"][j] for j in range(len(pred["massing"])) if j != i]}])
                intersection = base_mesh.intersection(disjoint_mesh).volume
                volume = base_mesh.volume
                iou = intersection/volume
                value += iou
                computed += 1

        result = value / computed if computed > 0 else 0
        return result