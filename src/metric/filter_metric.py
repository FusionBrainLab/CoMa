from abc import ABC, abstractmethod
from typing import Any, Dict, List
import json
import re

import pandas as pd

from .metric import Metric
from ..protocols import IndicatorProtocol

class FilterMetric(Metric):
    def __init__(self, *, base_metric: Metric,
                        filter: IndicatorProtocol) -> None:
        self.base_metric = base_metric
        self.filter = filter
    
    def __call__(self, *, submit: Dict[str, List[Any]]) -> float:
        samples = pd.DataFrame(submit)
        filtered_samples = samples[samples.apply(lambda row: self.filter(**row), axis=1)]
        if len(filtered_samples) == 0:
            raise
        filtered_samples = filtered_samples.to_dict("list")
        score = self.base_metric(submit=filtered_samples)
        return score