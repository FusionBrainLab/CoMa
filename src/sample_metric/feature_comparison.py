from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric

class FeatureComparison(SampleMetric):
    def __init__(self, *, pred_feature_metric: SampleMetric,
                        gt_feature_metric: SampleMetric) -> None:
        self.pred_feature_metric = pred_feature_metric
        self.gt_feature_metric = gt_feature_metric

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        pred_feature = self.pred_feature_metric(sample=sample)
        gt_feature = self.gt_feature_metric(sample=sample)
        feature_match = abs(gt_feature - pred_feature)/gt_feature
        return feature_match