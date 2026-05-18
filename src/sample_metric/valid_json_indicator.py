from abc import ABC, abstractmethod
from typing import Any, Dict, List
import json

from .sample_metric import SampleMetric

class ValidJsonIndicator(SampleMetric):
    def __init__(self, *, key: str) -> None:
        self.key = key
        
    def __call__(self, *, sample: Dict[str, Any]) -> float:
        try:
            json.loads(sample[self.key])
            score = 1
        except:
            score = 0
        return score