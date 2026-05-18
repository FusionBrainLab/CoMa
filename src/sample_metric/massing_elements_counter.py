from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

from .sample_metric import SampleMetric

class MassingElementsCounter(SampleMetric):
    def __init__(self, *, massing_key: str,
                        element: Literal["point", "polygon", "extrusion", "building"]) -> None:
        self.massing_key = massing_key
        self.element = element

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        massing = sample[self.massing_key]
        if self.element == "point":
            value = sum([1 for m in massing for e in m["massing"] for polygon in e["polygons"] for point in polygon])
        elif self.element == "polygon":
            value = sum([1 for m in massing for e in m["massing"] for polygon in e["polygons"]])
        elif self.element == "extrusion":
            value = sum([1 for m in massing for e in m["massing"]])
        elif self.element == "building":
            value = sum([1 for m in massing])
        else:
            raise
        return value