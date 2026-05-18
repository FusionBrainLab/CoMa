from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

import numpy as np

from .sample_metric import SampleMetric

class MassingBuildingsMetric(SampleMetric):
    def __init__(self, *, massing_key: str,
                        building_metric: SampleMetric,
                        building_key: str,
                        reduction: Literal["min", "max", "mean", "std"],
                        wrap_to_massing: bool) -> None:
        self.massing_key = massing_key
        self.building_metric = building_metric
        self.building_key = building_key
        self.reduction = reduction
        self.wrap_to_massing = wrap_to_massing

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        massing = sample[self.massing_key]
        values = []
        for m in massing:
            if self.wrap_to_massing:
                building = [m]
            else:
                building = m
            value = self.building_metric(sample={self.building_key:building})
            values.append(value)

        if self.reduction == "min":
            value = min(values)
        elif self.reduction == "max":
            value = max(values)
        elif self.reduction == "mean":
            value = np.mean(values)
        elif self.reduction == "std":
            value = np.std(values)
        else:
            raise
        return value