from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

from shapely import unary_union
import numpy as np

from .sample_metric import SampleMetric
from ..polygons_analyzer import PolygonsAnalyzer
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter

class MassingFootprintsMetric(SampleMetric):
    def __init__(self, *, massing_key: str,
                        polygons_metric: PolygonsAnalyzer,
                        reduction: Literal["min", "max", "mean", "std"]) -> None:
        self.massing_key = massing_key
        self.polygons_metric = polygons_metric
        self.reduction = reduction

        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        massing = sample[self.massing_key]
        values = []
        for m in massing:
            shapely_polygons = []
            for e in m["massing"]:
                polygons = [[(p[0], p[1]) for p in polygon] for polygon in e["polygons"]]
                shapely_polygons.append(self.polygons_to_shapely_converter(polygons=polygons))
            footprint = unary_union(shapely_polygons)
            footprint = self.shapely_to_polygons_converter(polygons=footprint)
            value = self.polygons_metric(polygons=footprint)
            values.append(value)

        if self.reduction == "min":
            value = min(values)
        elif self.reduction == "max":
            value = max(values)
        elif self.reduction == "mean":
            value = np.mean(values)
        elif self.reduction == "std":
            value = np.std(values)
        else:
            raise
        return value