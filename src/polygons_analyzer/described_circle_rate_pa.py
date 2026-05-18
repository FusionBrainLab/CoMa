from typing import List, Any, Dict, Literal, Tuple

import shapely

from .polygons_analyzer import PolygonsAnalyzer
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class DescribedCircleRatePA(PolygonsAnalyzer):
    def __init__(self) -> None:
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
    def __call__(self, *, polygons: List[List[Tuple[float, float]]]) -> float:
        shapely_polygons = self.polygons_to_shapely_converter(polygons=polygons)
        circle = shapely.minimum_bounding_circle(shapely_polygons)
        rate = shapely.area(shapely_polygons) / shapely.area(circle)
        return rate