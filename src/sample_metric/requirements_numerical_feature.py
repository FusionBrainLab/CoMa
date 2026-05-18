from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

from .sample_metric import SampleMetric

class RequirementsNumericalFeature(SampleMetric):
    def __init__(self, *, requirements_key: str,
                        feature_key: str,
                        reduction: Literal["min", "max", "mean"]) -> None:
        self.requirements_key = requirements_key
        self.feature_key = feature_key
        self.reduction = reduction

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        requirements = sample[self.requirements_key]
        features = [r[self.feature_key] for r in requirements]
        if self.reduction == "min":
            value = min(features)
        elif self.reduction == "max":
            value = max(features)
        elif self.reduction == "mean":
            value = sum(features)/len(features)
        else:
            raise
        return value