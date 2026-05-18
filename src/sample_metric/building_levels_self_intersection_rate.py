from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from shapely import unary_union
from shapely.geometry import MultiPolygon

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class BuildingLevelsSelfIntersectionRate(SampleMetric):
    def __init__(self, *, building_key: str) -> None:
        self.building_key = building_key
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        building = sample[self.building_key]
        
        values = []
        for i in range(len(building["massing"])):
            for j in range(i + 1, len(building["massing"])):
                e1 = building["massing"][i]
                e2 = building["massing"][j]
                p1 = self.polygons_to_shapely_converter(polygons=[[(point[0], point[1]) for point in polygon] for polygon in e1["polygons"]])
                p2 = self.polygons_to_shapely_converter(polygons=[[(point[0], point[1]) for point in polygon] for polygon in e2["polygons"]])

                if e1["bottom_elevation"] != e2["bottom_elevation"] or e1["top_elevation"] != e2["top_elevation"]:
                    continue
                else:
                    intersection = p1.intersection(p2)
                    union = p1.union(p2)
                    iou = intersection.area/union.area
                    values.append(iou)
        value = sum(values)/len(values)
        return value