from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

from .sample_metric import SampleMetric

class MassingComplexityMatch(SampleMetric):
    def __init__(self, *, gt_massing_key: str,
                        pred_massing_key: str,
                        complexity_type: Literal["point", "extrusion"]) -> None:
        self.gt_massing_key = gt_massing_key
        self.pred_massing_key = pred_massing_key
        self.complexity_type = complexity_type

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        gt_massing = sample[self.gt_massing_key]
        error = 0
        computed = 0
        for gt_building in gt_massing:
            pred_building = [m for m in sample[self.pred_massing_key] if m["id"] == gt_building["id"]]
            if len(pred_building) == 0:
                continue
            pred_building = pred_building[0]

            if self.complexity_type == "point":
                gt_complexity = sum([1 for e in gt_building["massing"] for polygon in e["polygons"] for point in polygon])
                pred_complexity = sum([1 for e in pred_building["massing"] for polygon in e["polygons"] for point in polygon])
            elif self.complexity_type == "extrusion":
                gt_complexity = sum([1 for e in gt_building["massing"]])
                pred_complexity = sum([1 for e in pred_building["massing"]])

            local_error = 1 - min(abs(gt_complexity - pred_complexity)/gt_complexity, 1)
            error += local_error
            computed += 1

        if computed == 0:
            raise
        
        error = error/computed
        return error