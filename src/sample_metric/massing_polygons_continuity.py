from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

import trimesh
import shapely
from trimesh import transformations
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class MassingPolygonsContinuity(SampleMetric):
    def __init__(self, *, pred_col: str) -> None:
        self.pred_col = pred_col
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        total_pred_massings = sample[self.pred_col]

        value = 0
        computed = 0
        for pred in total_pred_massings:
            for e in pred["massing"]:
                polygons = [[(point[0], point[1]) for point in polygon] for polygon in e["polygons"]]
                polygons = self.polygons_to_shapely_converter(polygons=polygons)
                if type(polygons) == MultiPolygon:
                    local_value = 1/len(polygons.geoms)
                else:
                    local_value = 1

                value += local_value
                computed += 1

        result = value / computed if computed > 0 else 0
        return result