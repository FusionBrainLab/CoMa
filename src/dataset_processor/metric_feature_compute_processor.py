from typing import List, Any, Dict
from functools import partial

import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel

from .dataset_processor import DatasetProcessor
from ..sample_metric import SampleMetric

class MetricFeatureComputeProcessor(DatasetProcessor):
    def __init__(self, *, feature_metrics: Dict[str, SampleMetric],
                        num_workers: int,
                        fail_value: Any,
                        drop_failed: bool) -> None:
        self.feature_metrics = feature_metrics
        self.num_workers = num_workers
        self.fail_value = fail_value
        self.drop_failed = drop_failed
        if self.num_workers > 1:
            pandarallel.initialize(nb_workers=self.num_workers, progress_bar=True)

    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        tqdm.pandas()
        for feature, metric in self.feature_metrics.items():
            print(feature)
            def compute_feature(row, metric):
                try:
                    return metric(sample=row.to_dict())
                except Exception:
                    return "failed"
            compute_feature = partial(compute_feature, metric=metric)
            if self.num_workers > 1:
                pd_dataset[feature] = pd_dataset.parallel_apply(compute_feature, axis=1)
            else:
                pd_dataset[feature] = pd_dataset.progress_apply(compute_feature, axis=1)
            if self.drop_failed:
                pd_dataset = pd_dataset[pd_dataset[feature] != "failed"]
            else:
                pd_dataset.loc[pd_dataset[feature] == "failed", feature] = self.fail_value
        return pd_dataset.to_dict("list")