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

class MassingFootprintShapeMatch(SampleMetric):
    def __init__(self, *, gt_massing_key: str,
                        pred_massing_key: str) -> None:
        self.gt_massing_key = gt_massing_key
        self.pred_massing_key = pred_massing_key
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        pred_massing = sample[self.pred_massing_key]
        gt_massing = sample[self.gt_massing_key]

        counter = 0
        sims = 0
        for gt in gt_massing:
            pred = [m for m in pred_massing if m["id"] == gt["id"]]
            if len(pred) == 0:
                continue
            pred = pred[0]

            pred_polygons = [[[(point[0], point[1]) for point in polygon] for polygon in e["polygons"]] for e in pred["massing"]]
            pred_footprint = shapely.unary_union([self.polygons_to_shapely_converter(polygons=p) for p in pred_polygons])
            if type(pred_footprint) == Polygon:
                pred_footprint = self.shapely_to_polygons_converter(polygons=Polygon(pred_footprint.exterior.coords))
            elif type(pred_footprint) == MultiPolygon:
                pred_footprint = self.shapely_to_polygons_converter(polygons=Polygon(pred_footprint.geoms[0].exterior.coords))
            pred_footprint = np.array(pred_footprint[0])

            gt_polygons = [[[(point[0], point[1]) for point in polygon] for polygon in e["polygons"]] for e in gt["massing"]]
            gt_footprint = shapely.unary_union([self.polygons_to_shapely_converter(polygons=p) for p in gt_polygons])
            if type(gt_footprint) == Polygon:
                gt_footprint = self.shapely_to_polygons_converter(polygons=Polygon(gt_footprint.exterior.coords))
            elif type(gt_footprint) == MultiPolygon:
                gt_footprint = self.shapely_to_polygons_converter(polygons=Polygon(gt_footprint.geoms[0].exterior.coords))
            gt_footprint = np.array(gt_footprint[0])

            sim = shape_similarity(pred_footprint, gt_footprint)
            sims += sim
            counter += 1
        
        output = sims/counter if counter > 0 else 0
        return output
