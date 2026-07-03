from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

import numpy as np

from .sample_metric import SampleMetric

class MassingExtrusionHeightDiversity(SampleMetric):
    def __init__(self, *, massing_key: str,
                        diversity_type: Literal["std", "unique"]) -> None:
        self.massing_key = massing_key
        self.diversity_type = diversity_type

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        massing = sample[self.massing_key]

        values = []
        for m in massing:
            for e in m["massing"]:
                height = e["top_elevation"] - e["bottom_elevation"]
                values.append(height)
        
        if self.diversity_type == "std":
            value = np.std(values)/np.mean(values)
        elif self.diversity_type == "unique":
            value = len(list(set(values)))/len(values)
        else:
            raise
        return value