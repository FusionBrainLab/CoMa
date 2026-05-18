from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

from .sample_metric import SampleMetric

class RequirementsIdCount(SampleMetric):
    def __init__(self, *, requirements_key: str) -> None:
        self.requirements_key = requirements_key

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        requirements = sample[self.requirements_key]
        value = len(requirements)
        return value