from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

import shapely

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class BuildingTotalArea(SampleMetric):
    def __init__(self, *, building_key: str,
                        floor_height: float) -> None:
        self.building_key = building_key
        self.floor_height = floor_height
        self.polygons_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        building = sample[self.building_key]["massing"]

        min_elevation = min([e["bottom_elevation"] for e in building])
        max_elevation = max([e["top_elevation"] for e in building])
        height = max_elevation - min_elevation
        n_floors = math.ceil(height / self.floor_height)

        area = 0
        heights = [min_elevation] + [self.floor_height * (i + 1) for i in range(n_floors - 1)] if n_floors > 1 else [min_elevation]
        for local_height in heights:
            local_footprints = [e["polygons"] for e in building if local_height >= e["bottom_elevation"] and local_height < e["top_elevation"]]
            local_footprints = [[[(p[0], p[1]) for p in polygon] for polygon in foot] for foot in local_footprints]
            local_footprints = [self.polygons_converter(polygons=p) for p in local_footprints]
            if len(local_footprints) == 0:
                continue
            local_floor = local_footprints[0]
            for i in range(1, len(local_footprints)):
                local_floor = local_floor.union(local_footprints[i])
            local_area = shapely.area(local_floor)
            area += local_area
        return round(area)
