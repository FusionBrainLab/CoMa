from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

import numpy as np
from shapely.geometry import Polygon, Point

from .sample_metric import SampleMetric

class CornersSoftness(SampleMetric):
    def __init__(self, *, massing_col: str,
                        min_soft_angle: float,
                        min_sharp_angle: float,
                        compute_type: Literal["mean", "sharp_only"]) -> None:
        self.massing_col = massing_col
        self.min_soft_angle = min_soft_angle
        self.min_sharp_angle = min_sharp_angle
        self.compute_type = compute_type

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        pred_massing = sample[self.massing_col]
        
        coefs = []
        count = 0
        for m in pred_massing:
            for e in m["massing"]:
                for polygon_coords in e["polygons"]:
                    coefficients = []
                    n = len(polygon_coords)
                    for i in range(len(polygon_coords)):
                        p1 = Point(polygon_coords[i-1])
                        p2 = Point(polygon_coords[i])
                        p3 = Point(polygon_coords[(i+1)%n])
                        
                        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
                        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
                        
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        
                        angle_deg = np.degrees(np.arccos(cos_angle))
                        
                        min_angle = self.min_sharp_angle
                        max_angle = self.min_soft_angle
                        if angle_deg <= min_angle:
                            coeff = 0.0
                        elif angle_deg >= max_angle:
                            coeff = 1.0
                        else:
                            coeff = (angle_deg - min_angle) / (max_angle - min_angle)

                        if self.compute_type == "mean":
                            coefs.append(coeff)
                            count += 1
                        elif self.compute_type == "sharp_only":
                            if angle_deg < max_angle:
                                coefs.append(coeff)
                                count += 1

        result = sum(coefs) / count if count > 0 else 1
        if str(result) == "nan":
            result = 0
        return result
