from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric

class ReverseMetric(SampleMetric):
    def __init__(self, *, base_metric: SampleMetric,
                        max_value: float) -> None:
        self.base_metric = base_metric
        self.max_value = max_value

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        return self.max_value - self.base_metric(sample=sample)