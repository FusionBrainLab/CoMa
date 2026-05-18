from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from shapely import unary_union
from shapely.geometry import MultiPolygon

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class BuildingFootprintContinuity(SampleMetric):
    def __init__(self, *, building_key: str) -> None:
        self.building_key = building_key
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        building = sample[self.building_key]
        
        shapely_polygons = []
        for e in building["massing"]:
            polygons = [[(point[0], point[1]) for point in polygon] for polygon in e["polygons"]]
            polygons = self.polygons_to_shapely_converter(polygons=polygons)
            shapely_polygons.append(polygons)
        footprint = unary_union(shapely_polygons)
        n_parts = 1
        if type(footprint) == MultiPolygon:
            n_parts = len(footprint.geoms)
        value = 1/n_parts
        return value