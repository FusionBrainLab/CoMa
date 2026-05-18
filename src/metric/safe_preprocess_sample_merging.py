from abc import ABC, abstractmethod
from typing import Any, Dict, List
import json
import re

import pandas as pd
from tqdm import tqdm

from .metric import Metric
from ..sample_metric import SampleMetric
from ..floats_merger import FloatsMerger
from ..core.base import Function

class SafePreprocessSampleMerging(Metric):
    def __init__(self, *, sample_metric: SampleMetric,
                        results_merger: FloatsMerger,
                        preprocessor: Function,
                        default_sample_result: float) -> None:
        self.sample_metric = sample_metric
        self.results_merger = results_merger
        self.preprocessor = preprocessor
        self.default_sample_result = default_sample_result
    
    def __call__(self, *, submit: Dict[str, List[Any]]) -> float:
        samples = pd.DataFrame(submit)
        tqdm.pandas()
        def validate(sample):
            """try:
                processed_sample = self.preprocessor(**sample)
            except:
                return self.default_sample_result
            local_result = self.sample_metric(sample=processed_sample)
            return local_result"""
            try:
                processed_sample = self.preprocessor(**sample)
                local_result = self.sample_metric(sample=processed_sample)
                return local_result
            except:
                return self.default_sample_result
        samples["metric"] = samples.progress_apply(lambda row: validate(row), axis=1)
        results = samples["metric"].values.tolist()
        score = self.results_merger(floats=results)
        return score