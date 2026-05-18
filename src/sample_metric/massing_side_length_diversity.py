from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

import shapely
from shapely import maximum_inscribed_circle, minimum_rotated_rectangle
from shapesimilarity import shape_similarity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
import numpy as np

from .sample_metric import SampleMetric

class MassingSideLengthDiversity(SampleMetric):
    def __init__(self, *, massing_key: str,
                        diversity_type: Literal["std", "unique"]) -> None:
        self.massing_key = massing_key
        self.diversity_type = diversity_type

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        massing = sample[self.massing_key]

        values = []
        for m in massing:
            for e in m["massing"]:
                for polygon_coords in e["polygons"]:
                    n = len(polygon_coords)
                    for i in range(len(polygon_coords)):
                        p1 = Point(polygon_coords[i])
                        p2 = Point(polygon_coords[(i+1)%n])
                        
                        v = np.array([p1.x - p2.x, p1.y - p2.y])
                        length = round(np.linalg.norm(v), 2)
                        
                        values.append(length)
        
        if self.diversity_type == "std":
            value = np.std(values)/np.mean(values)
        elif self.diversity_type == "unique":
            value = len(list(set(values)))/len(values)
        else:
            raise
        return value