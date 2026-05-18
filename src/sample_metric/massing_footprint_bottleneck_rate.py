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

class MassingFootprintBottleneckRate(SampleMetric):
    def __init__(self, *, pred_massing_key: str,
                        max_edge_chunk: float,
                        max_bottleneck: float) -> None:
        self.pred_massing_key = pred_massing_key
        self.max_edge_chunk = max_edge_chunk
        self.max_bottleneck = max_bottleneck
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        pred_massing = sample[self.pred_massing_key]

        counter = 0
        values = 0
        for pred in pred_massing:
            pred_polygons = [[[(point[0], point[1]) for point in polygon] for polygon in e["polygons"]] for e in pred["massing"]]
            pred_footprint = shapely.unary_union([self.polygons_to_shapely_converter(polygons=p) for p in pred_polygons])
            footprint_polygons = self.shapely_to_polygons_converter(polygons=pred_footprint)

            boundary = pred_footprint.boundary
            
            length = boundary.length
            num_points = int(np.ceil(length / self.max_edge_chunk))
            points = [boundary.interpolate(i * length / num_points) for i in range(num_points)]
            
            radiuses = []
            for point in points:
                min_dist = float('inf')
                for polygon in footprint_polygons:
                    for i in range(len(polygon)):
                        p1 = polygon[i]
                        p2 = polygon[(i + 1) % len(polygon)]
                        
                        edge = LineString([p1, p2])
                        if edge.distance(point) < 0.001:
                            continue
                        
                        dist = edge.distance(point)
                        if dist < min_dist:
                            min_dist = dist
                
                radiuses.append(min_dist)

            rate = len([r for r in radiuses if r <= self.max_bottleneck]) / len(radiuses)
            values += rate
            counter += 1
        
        if counter == 0:
            raise
        
        output = values/counter
        return output