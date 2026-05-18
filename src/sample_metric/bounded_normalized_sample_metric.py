from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric

class BoundedNormalizedSampleMetric(SampleMetric):
    def __init__(self, *, base_metric: SampleMetric,
                        old_lower_bound: float,
                        old_upper_bound: float,
                        new_lower_bound: float,
                        new_upper_bound: float) -> None:
        self.base_metric = base_metric
        self.old_lower_bound = old_lower_bound
        self.old_upper_bound = old_upper_bound
        self.new_lower_bound = new_lower_bound
        self.new_upper_bound = new_upper_bound

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        value = self.base_metric(sample=sample)
        if value < self.old_lower_bound:
            bounded_value = self.old_lower_bound
        elif value > self.old_upper_bound:
            bounded_value = self.old_upper_bound
        else:
            bounded_value = value
        
        result = self.new_lower_bound + (bounded_value - self.old_lower_bound) * (self.new_upper_bound - self.new_lower_bound) / (self.old_upper_bound - self.old_lower_bound)
        return result