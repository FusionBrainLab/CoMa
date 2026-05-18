from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric

class FeatureMatch(SampleMetric):
    def __init__(self, *, feature_metric: SampleMetric,
                        gt_feature_key: str) -> None:
        self.feature_metric = feature_metric
        self.gt_feature_key = gt_feature_key

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        pred_feature = self.feature_metric(sample=sample)
        gt_feature = sample[self.gt_feature_key]
        feature_match = abs(gt_feature - pred_feature)/gt_feature
        return feature_match