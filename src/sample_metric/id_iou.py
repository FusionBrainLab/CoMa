from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric

class IdIoU(SampleMetric):
    def __init__(self, *, output_col: str) -> None:
        self.output_col = output_col

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        pred_massing = sample[self.output_col]
        pred_ids = set([m["id"] for m in pred_massing])

        gt_requirements = sample["requirements"]
        gt_ids = set([m["id"] for m in gt_requirements])
        
        iou = len(pred_ids.intersection(gt_ids))/len(pred_ids.union(gt_ids))
        return iou