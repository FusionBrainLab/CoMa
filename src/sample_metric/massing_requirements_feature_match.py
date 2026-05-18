from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric
from .feature_match import FeatureMatch

class MassingRequirementsFeatureMatch(SampleMetric):
    def __init__(self, *, feature_metric: SampleMetric,
                        massing_key: str,
                        requirements_key: str,
                        requirements_feature_key: str,
                        building_key: str,
                        feature_key: str) -> None:
        self.feature_metric = feature_metric
        self.massing_key = massing_key
        self.requirements_key = requirements_key
        self.requirements_feature_key = requirements_feature_key
        self.building_key = building_key
        self.feature_key = feature_key

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        total_gt_requirements = sample[self.requirements_key]
        value = 0
        computed = 0
        for gt_requirements in total_gt_requirements:
            gt_feature = gt_requirements[self.requirements_feature_key]
            pred_massing = [m for m in sample[self.massing_key] if str(m["id"]) == str(gt_requirements["id"])]
            if len(pred_massing) == 0:
                continue
            pred_massing = pred_massing[0]

            match_value = self.feature_metric(sample={
                self.building_key: pred_massing,
                self.feature_key: gt_feature
            })
            
            value += match_value
            computed += 1

        if computed == 0:
            raise
        
        value = value/computed
        return value