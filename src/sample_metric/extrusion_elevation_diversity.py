from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

import shapely
from shapely import maximum_inscribed_circle, minimum_rotated_rectangle
from shapesimilarity import shape_similarity
from shapely.geometry import Polygon, MultiPolygon, LineString
import numpy as np

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter

class ExtrusionElevationDiversity(SampleMetric):
    def __init__(self, *, pred_massing_key: str) -> None:
        self.pred_massing_key = pred_massing_key
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        pred_massing = sample[self.pred_massing_key]

        counter = 0
        values = 0
        for pred in pred_massing:
            elevations = [e["top_elevation"] for e in pred["massing"]]
            rate = len(list(set(elevations))) / len(elevations)
            values += rate
            counter += 1
        
        output = values/counter if counter > 0 else 0
        return output