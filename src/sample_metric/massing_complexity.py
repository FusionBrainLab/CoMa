from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

from .sample_metric import SampleMetric

class MassingComplexity(SampleMetric):
    def __init__(self, *, pred_massing_key: str,
                        complexity_type: Literal["point", "polygon", "extrusion"]) -> None:
        self.pred_massing_key = pred_massing_key
        self.complexity_type = complexity_type

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        values = 0
        computed = 0
        for pred_building in sample[self.pred_massing_key]:
            if self.complexity_type == "point":
                pred_complexity = sum([1 for e in pred_building["massing"] for polygon in e["polygons"] for point in polygon])
            elif self.complexity_type == "polygon":
                pred_complexity = sum([1 for e in pred_building["massing"] for polygon in e["polygons"]])
            elif self.complexity_type == "extrusion":
                pred_complexity = sum([1 for e in pred_building["massing"]])

            values += pred_complexity
            computed += 1
        
        value = values/computed if computed > 0 else 0
        return values