from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from shapely import unary_union
from shapely.geometry import MultiPolygon

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class BuildingExtrusionsContinuity(SampleMetric):
    def __init__(self, *, building_key: str) -> None:
        self.building_key = building_key
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        building = sample[self.building_key]
        
        values = []
        for e in building["massing"]:
            polygons = [[(point[0], point[1]) for point in polygon] for polygon in e["polygons"]]
            polygons = self.polygons_to_shapely_converter(polygons=polygons)
            n_parts = 1
            if type(polygons) == MultiPolygon:
                n_parts = len(polygons.geoms)
            value = 1/n_parts
            values.append(value)

        value = sum(values)/len(values)
        return value