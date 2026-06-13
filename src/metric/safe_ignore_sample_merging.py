from abc import ABC, abstractmethod
from typing import Any, Dict, List
import json
import re
import traceback

import pandas as pd
from tqdm import tqdm

from .metric import Metric
from ..sample_metric import SampleMetric
from ..floats_merger import FloatsMerger
from ..core.base import Function

class SafeIgnoreSampleMerging(Metric):
    def __init__(self, *, sample_metric: SampleMetric,
                        results_merger: FloatsMerger) -> None:
        self.sample_metric = sample_metric
        self.results_merger = results_merger
    
    def __call__(self, *, submit: Dict[str, List[Any]]) -> float:
        samples = pd.DataFrame(submit)
        results = []
        for ind, row in tqdm(samples.iterrows(), total=len(samples)):
            try:
                local_result = self.sample_metric(sample=row.to_dict())
                results.append(local_result)
            except Exception as e:
                """print(f"Error processing sample {ind}: {e}")
                print(traceback.format_exc())"""
                continue
        score = self.results_merger(floats=results)
        return score