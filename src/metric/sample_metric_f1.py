from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from .metric import Metric
from ..sample_metric import SampleMetric


class SampleMetricF1(Metric):
    """Evaluate a SampleMetric as a binary classifier using optimal-threshold F1.

    Finds the score threshold that maximises F1 on the full set and reports that
    F1 together with the corresponding precision, recall, and threshold.
    Returns the best F1 score as the single float (for easy comparison).
    """

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
            except Exception:
                return None

        samples["pred_score"] = samples.progress_apply(get_metric, axis=1)
        samples = samples[samples["pred_score"].notna()]

        scores = samples["pred_score"].to_numpy(dtype=float)
        labels = samples[self.gt_label_key].to_numpy()

        # sweep thresholds over the score distribution
        thresholds = np.unique(scores)
        best_f1 = 0.0
        for thr in thresholds:
            preds = (scores >= thr).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1

        return float(best_f1)
