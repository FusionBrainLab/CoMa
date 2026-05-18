from abc import ABC, abstractmethod
from typing import List, Any, Dict, Union, Tuple

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import shapely

from .core.base import Function

class PolygonsToShapelyConverter(Function):
    def __call__(self, *, polygons: List[List[Tuple[float, float]]]) -> shapely.Geometry:
        shapely_cur_polygon = Polygon(polygons[0])
        for i in range(1, len(polygons)):
            poly = Polygon(polygons[i])
            if shapely_cur_polygon.equals(poly):
                continue
            elif shapely_cur_polygon.contains_properly(poly):
                shapely_cur_polygon = shapely_cur_polygon.difference(poly)
            elif poly.contains_properly(shapely_cur_polygon):
                shapely_cur_polygon = poly.difference(shapely_cur_polygon)
            else:
                shapely_cur_polygon = shapely_cur_polygon.union(poly)
        return shapely_cur_polygon