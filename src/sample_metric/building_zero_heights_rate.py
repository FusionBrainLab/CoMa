from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

import numpy as np

from .sample_metric import SampleMetric

class BuildingZeroHeightsRate(SampleMetric):
    def __init__(self, *, building_key: str) -> None:
        self.building_key = building_key

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        building = sample[self.building_key]
        
        values = []
        for e in building["massing"]:
            if e["top_elevation"] - e["bottom_elevation"] == 0:
                values.append(1)
            else:
                values.append(0)
        value = sum(values)/len(values)
        return value