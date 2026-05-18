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

class SafePreprocessIgnoreSampleMerging(Metric):
    def __init__(self, *, sample_metric: SampleMetric,
                        results_merger: FloatsMerger,
                        preprocessor: Function) -> None:
        self.sample_metric = sample_metric
        self.results_merger = results_merger
        self.preprocessor = preprocessor
    
    def __call__(self, *, submit: Dict[str, List[Any]]) -> float:
        samples = pd.DataFrame(submit)
        results = []
        for ind, row in tqdm(samples.iterrows(), total=len(samples)):
            try:
                processed_sample = self.preprocessor(**row.to_dict())
                local_result = self.sample_metric(sample=processed_sample)
                results.append(local_result)
            except:
                #raise
                continue
        score = self.results_merger(floats=results)
        return score