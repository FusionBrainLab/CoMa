from abc import ABC, abstractmethod
from typing import Any, Dict, List
import json
import re

import pandas as pd
from tqdm import tqdm

from .metric import Metric
from ..sample_metric import SampleMetric
from ..floats_merger import FloatsMerger

class SampleMerging(Metric):
    def __init__(self, *, sample_metric: SampleMetric,
                        results_merger: FloatsMerger) -> None:
        self.sample_metric = sample_metric
        self.results_merger = results_merger
    
    def __call__(self, *, submit: Dict[str, List[Any]]) -> float:
        """samples = [{k: submit[k][i] for k in submit.keys()} for i in range(len(submit[list(submit.keys())[0]]))]
        for i, sample in tqdm(enumerate(samples), total=len(samples)):
            print(i)
            value = self.sample_metric(sample=sample)"""
        samples = pd.DataFrame(submit)
        tqdm.pandas()
        samples["metric"] = samples.progress_apply(lambda row: self.sample_metric(sample=row.to_dict()), axis=1)
        results = samples["metric"].values.tolist()
        score = self.results_merger(floats=results)
        return score