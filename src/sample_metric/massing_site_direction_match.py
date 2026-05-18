from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

import shapely
from shapely import maximum_inscribed_circle, minimum_rotated_rectangle
import numpy as np

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

def get_polygon_elongation_direction(polygon):
    min_rect = polygon.minimum_rotated_rectangle
    coords = list(min_rect.exterior.coords)[:-1]
    
    distances = []
    for i in range(len(coords)):
        p1 = coords[i]
        p2 = coords[(i+1) % len(coords)]
        dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        distances.append(dist)
    
    longest_side_idx = np.argmax(distances)
    p1 = coords[longest_side_idx]
    p2 = coords[(longest_side_idx+1) % len(coords)]
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    angle_deg = angle_deg % 180
    return angle_deg

class MassingSiteDirectionMatch(SampleMetric):
    def __init__(self, *, massing_key: str,
                        site_key: str) -> None:
        self.massing_key = massing_key
        self.site_key = site_key
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        pred_massing = sample[self.massing_key]

        if len(pred_massing) == 1:
            polygons = [[[(point[0], point[1]) for point in polygon] for polygon in e["polygons"]] for e in pred_massing[0]["massing"]]
            footprint = shapely.unary_union([self.polygons_to_shapely_converter(polygons=p) for p in polygons])
        else:
            polygons = [[[(point[0], point[1]) for point in polygon] for polygon in e["polygons"]] for m in pred_massing for e in m["massing"]]
            footprint = shapely.unary_union([self.polygons_to_shapely_converter(polygons=p) for p in polygons])
            footprint = shapely.convex_hull(footprint)

        site = sample[self.site_key]
        site = [[(point[0], point[1]) for point in polygon] for polygon in site]
        site = self.polygons_to_shapely_converter(polygons=site)

        massing_dir = get_polygon_elongation_direction(footprint)
        site_dir = get_polygon_elongation_direction(site)

        dif = abs(massing_dir - site_dir)
        if dif > 90:
            dif = 180 - dif
        result = 1 - min(dif/90, 1)
        return result