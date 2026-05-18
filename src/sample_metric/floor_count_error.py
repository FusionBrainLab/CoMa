from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric

class FloorCountError(SampleMetric):
    def __init__(self, *, output_col: str,
                        floor_height_key: str) -> None:
        self.output_col = output_col
        self.floor_height_key = floor_height_key

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        total_gt_requirements = sample["requirements"]
        error = 0
        computed = 0
        for gt_requirements in total_gt_requirements:
            gt_n_floors = gt_requirements["n_floors"]
            gt_floor_height = gt_requirements[self.floor_height_key]

            pred_massing = [m for m in sample[self.output_col] if str(m["id"]) == str(gt_requirements["id"])]
            if len(pred_massing) == 0:
                continue
            pred_massing = pred_massing[0]["massing"]
            min_elevation = min([e["bottom_elevation"] for e in pred_massing])
            max_elevation = max([e["top_elevation"] for e in pred_massing])
            pred_height = max_elevation - min_elevation

            pred_n_floors = math.ceil(pred_height / gt_floor_height)

            local_error = abs(gt_n_floors - pred_n_floors)/gt_n_floors
            error += local_error
            computed += 1

        if computed == 0:
            raise
        
        error = error/computed
        return error
        