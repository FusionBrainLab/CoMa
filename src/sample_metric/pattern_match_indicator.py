from abc import ABC, abstractmethod
from typing import Any, Dict, List
import json
import re

from .sample_metric import SampleMetric

class PatternMatchIndicator(SampleMetric):
    def __init__(self, *, key: str, pattern: str) -> None:
        self.key = key
        self.pattern = pattern
    
    def __call__(self, *, sample: Dict[str, Any]) -> float:
        matches = re.findall(self.pattern, sample[self.key])
        if len(matches) > 0:
            score = 1
        else:
            score = 0
        return score