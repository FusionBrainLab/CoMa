from abc import ABC, abstractmethod
from typing import List, Any, Dict, Union, Tuple

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
import shapely

from .core.base import Function

class PointToShapelyConverter(Function):
    def __call__(self, *, point: Tuple[float, float]) -> shapely.Geometry:
        return Point(point)