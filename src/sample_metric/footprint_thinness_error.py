from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class FootprintThinnessError(SampleMetric):
    def __init__(self, *, gt_massing_key: str,
                        pred_massing_key: str) -> None:
        self.gt_massing_key = gt_massing_key
        self.pred_massing_key = pred_massing_key
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        gt_massing = sample[self.gt_massing_key]
        error = 0
        computed = 0
        for gt_building in gt_massing:
            pred_building = [m for m in sample[self.pred_massing_key] if m["id"] == gt_building["id"]]
            if len(pred_building) == 0:
                continue
            pred_building = pred_building[0]

            try:
                shapely_polygons = []
                for e in gt_building["massing"]:
                    polygons = [[(p[0], p[1]) for p in polygon] for polygon in e["polygons"]]
                    shapely_polygons.append(self.polygons_to_shapely_converter(polygons=polygons))
                gt_footprint = unary_union(shapely_polygons)

                shapely_polygons = []
                for e in pred_building["massing"]:
                    polygons = [[(p[0], p[1]) for p in polygon] for polygon in e["polygons"]]
                    shapely_polygons.append(self.polygons_to_shapely_converter(polygons=polygons))
                pred_footprint = unary_union(shapely_polygons)

                gt_thinness = 4 * math.pi * shapely.area(gt_footprint) / (shapely.length(gt_footprint) ** 2)
                pred_thinness = 4 * math.pi * shapely.area(pred_footprint) / (shapely.length(pred_footprint) ** 2)

            except:
                raise
                continue

            local_error = min(abs(gt_thinness - pred_thinness)/gt_thinness, 1)
            error += local_error
            computed += 1

        if computed == 0:
            raise
        
        error = error/computed
        return error