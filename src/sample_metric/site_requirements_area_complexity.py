from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

from .sample_metric import SampleMetric
from ..polygons_analyzer import PolygonsAnalyzer
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter

class SiteRequirementsAreaComplexity(SampleMetric):
    def __init__(self, *, site_key: str,
                        requirements_key: str,
                        total_area_key: str,
                        n_floors_key: str) -> None:
        self.site_key = site_key
        self.requirements_key = requirements_key
        self.total_area_key = total_area_key
        self.n_floors_key = n_floors_key

        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        site = sample[self.site_key]
        site = [[(p[0], p[1]) for p in polygon] for polygon in site]
        site = self.polygons_to_shapely_converter(polygons=site)
        requirements = sample[self.requirements_key]

        min_total_area = 0
        for r in requirements:
            total_area = r[self.total_area_key]
            floors_count = r[self.n_floors_key]
            min_footprint_area = total_area / floors_count
            min_total_area += min_footprint_area
        complexity = site.area/min_total_area
        return complexity