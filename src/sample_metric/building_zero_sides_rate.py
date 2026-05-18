from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

import numpy as np

from .sample_metric import SampleMetric

class BuildingZeroSidesRate(SampleMetric):
    def __init__(self, *, building_key: str) -> None:
        self.building_key = building_key

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        building = sample[self.building_key]
        
        values = []
        for e in building["massing"]:
            for polygon in e["polygons"]:
                for i in range(1, len(polygon)):
                    if polygon[i][0] == polygon[i-1][0] and polygon[i][1] == polygon[i-1][1]:
                        values.append(0)
                    else:
                        values.append(1)
        value = sum(values)/len(values)
        return value