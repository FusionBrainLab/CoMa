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

class MassingSideLengthError(SampleMetric):
    def __init__(self, *, pred_massing_key: str,
                        gt_massing_key) -> None:
        self.pred_massing_key = pred_massing_key
        self.gt_massing_key = gt_massing_key

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        pred_massing = sample[self.pred_massing_key]
        gt_massing = sample[self.gt_massing_key]

        values = 0
        counter = 0
        for gt in gt_massing:
            pred = [m for m in pred_massing if m["id"] == gt["id"]]
            if len(pred) == 0:
                continue
            pred = pred[0]

            local_results = []
            for massing in [pred, gt]:
                local_values = 0
                local_counter = 0
                for e in massing["massing"]:
                    for polygon_coords in e["polygons"]:
                        n = len(polygon_coords)
                        for i in range(len(polygon_coords)):
                            p1 = Point(polygon_coords[i])
                            p2 = Point(polygon_coords[(i+1)%n])
                            
                            v = np.array([p1.x - p2.x, p1.y - p2.y])
                            length = np.linalg.norm(v)
                            
                            local_values += length
                            local_counter += 1
                local_result = local_values / local_counter if local_counter > 0 else 0
                local_results.append(local_result)

            error = abs(local_results[1] - local_results[0])/local_results[1]
            values += error
            counter += 1
        
        output = values / counter if counter > 0 else 0
        return output