from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric

class BoundedSampleMetric(SampleMetric):
    def __init__(self, *, base_metric: SampleMetric,
                        lower_bound: float,
                        upper_bound: float) -> None:
        self.base_metric = base_metric
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        value = self.base_metric(sample=sample)
        if value < self.lower_bound:
            return self.lower_bound
        elif value > self.upper_bound:
            return self.upper_bound
        else:
            return value
