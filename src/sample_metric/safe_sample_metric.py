from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric

class SafeSampleMetric(SampleMetric):
    def __init__(self, *, base_metric: SampleMetric,
                        default_result: Any) -> None:
        self.base_metric = base_metric
        self.default_result = default_result

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        try:
            return self.base_metric(sample=sample)
        except:
            return self.default_result