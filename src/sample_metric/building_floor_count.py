from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric

class BuildingFloorCount(SampleMetric):
    def __init__(self, *, building_key: str,
                        floor_height: float) -> None:
        self.building_key = building_key
        self.floor_height = floor_height

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        building = sample[self.building_key]["massing"]
        min_elevation = min([e["bottom_elevation"] for e in building])
        max_elevation = max([e["top_elevation"] for e in building])
        height = max_elevation - min_elevation
        n_floors = math.ceil(height / self.floor_height)
        return n_floors