from typing import List, Any, Dict, Literal, Tuple
import math

import shapely

from .polygons_analyzer import PolygonsAnalyzer
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class InscribedCircleRatePA(PolygonsAnalyzer):
    def __init__(self) -> None:
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
    def __call__(self, *, polygons: List[List[Tuple[float, float]]]) -> float:
        shapely_polygons = self.polygons_to_shapely_converter(polygons=polygons)
        circle = shapely.maximum_inscribed_circle(shapely_polygons)
        circle_area = math.pi*circle.length**2
        rate = circle_area / shapely.area(shapely_polygons)
        return rate