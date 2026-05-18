from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric
from ..polygons_analyzer import PolygonsAnalyzer
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter

class SitePolygonMetric(SampleMetric):
    def __init__(self, *, site_key: str,
                        polygons_metric: PolygonsAnalyzer) -> None:
        self.site_key = site_key
        self.polygons_metric = polygons_metric

        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        site = sample[self.site_key]
        site = [[(p[0], p[1]) for p in polygon] for polygon in site]
        value = self.polygons_metric(polygons=site)
        return value
