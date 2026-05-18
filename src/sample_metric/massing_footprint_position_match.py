from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

import shapely
from shapely import maximum_inscribed_circle, minimum_rotated_rectangle
from shapesimilarity import shape_similarity
from shapely.geometry import Polygon, MultiPolygon
import numpy as np

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter

class MassingFootprintPositionMatch(SampleMetric):
    def __init__(self, *, gt_massing_key: str,
                        pred_massing_key: str,
                        site_key: str) -> None:
        self.gt_massing_key = gt_massing_key
        self.pred_massing_key = pred_massing_key
        self.site_key = site_key
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        pred_massing = sample[self.pred_massing_key]
        gt_massing = sample[self.gt_massing_key]
        site_contour = [[(point[0], point[1]) for point in polygon] for polygon in sample[self.site_key]]
        site_contour = self.polygons_to_shapely_converter(polygons=site_contour)

        min_rect = site_contour.minimum_rotated_rectangle
        coords = list(min_rect.exterior.coords)[:-1]
        distances = []
        for i in range(len(coords)):
            p1 = coords[i]
            p2 = coords[(i+1) % len(coords)]
            dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            distances.append(dist)
        max_side = max(distances)

        counter = 0
        values = 0
        for gt in gt_massing:
            pred = [m for m in pred_massing if m["id"] == gt["id"]]
            if len(pred) == 0:
                continue
            pred = pred[0]

            pred_polygons = [[[(point[0], point[1]) for point in polygon] for polygon in e["polygons"]] for e in pred["massing"]]
            pred_footprint = shapely.unary_union([self.polygons_to_shapely_converter(polygons=p) for p in pred_polygons])

            gt_polygons = [[[(point[0], point[1]) for point in polygon] for polygon in e["polygons"]] for e in gt["massing"]]
            gt_footprint = shapely.unary_union([self.polygons_to_shapely_converter(polygons=p) for p in gt_polygons])

            pred_centroid = shapely.centroid(pred_footprint)
            gt_centroid = shapely.centroid(gt_footprint)
            
            value = 1 - pred_centroid.distance(gt_centroid)/max_side
            values += value
            counter += 1
        
        output = values/counter if counter > 0 else 0
        return output