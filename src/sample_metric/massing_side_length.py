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

class MassingSideLength(SampleMetric):
    def __init__(self, *, pred_massing_key: str) -> None:
        self.pred_massing_key = pred_massing_key

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        pred_massing = sample[self.pred_massing_key]

        values = 0
        counter = 0
        for m in pred_massing:
            for e in m["massing"]:
                for polygon_coords in e["polygons"]:
                    n = len(polygon_coords)
                    for i in range(len(polygon_coords)):
                        p1 = Point(polygon_coords[i])
                        p2 = Point(polygon_coords[(i+1)%n])
                        
                        v = np.array([p1.x - p2.x, p1.y - p2.y])
                        length = np.linalg.norm(v)
                        
                        values += length
                        counter += 1
        
        output = values / counter if counter > 0 else 0
        return output