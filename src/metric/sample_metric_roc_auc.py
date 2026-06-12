from abc import ABC, abstractmethod
from typing import Any, Dict, List
import json
import re

import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .metric import Metric
from ..sample_metric import SampleMetric

class SampleMetricROCAUC(Metric):
    def __init__(self, *, sample_metric: SampleMetric,
                        gt_label_key: str) -> None:
        self.sample_metric = sample_metric
        self.gt_label_key = gt_label_key
    
    def __call__(self, *, submit: Dict[str, List[Any]]) -> float:
        tqdm.pandas()
        samples = pd.DataFrame(submit)
        def get_metric(row):
            try:
                return self.sample_metric(sample=row.to_dict())
            except Exception as e:
                return None
        samples["pred_labels"] = samples.progress_apply(get_metric, axis=1)
        samples = samples[samples["pred_labels"].notna()]
        roc_auc = roc_auc_score(samples[self.gt_label_key], samples["pred_labels"])
        return roc_auc