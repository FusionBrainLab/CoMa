from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric
from ..core.base import Function

class FunctionExceptionIndicator(SampleMetric):
    def __init__(self, *, function: Function) -> None:
        self.function = function

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        try:
            self.function(**sample)
            return 1
        except:
            return 0