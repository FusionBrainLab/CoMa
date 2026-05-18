from typing import List, Any, Dict, Literal, Tuple
import math

import shapely
import numpy as np

from .polygons_analyzer import PolygonsAnalyzer
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class CircularityPA(PolygonsAnalyzer):
    def __init__(self) -> None:
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
    def __call__(self, *, polygons: List[List[Tuple[float, float]]]) -> float:
        shapely_polygons = self.polygons_to_shapely_converter(polygons=polygons)
        area = shapely_polygons.area
        perimeter = shapely_polygons.length
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        return circularity