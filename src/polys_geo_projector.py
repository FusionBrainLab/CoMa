from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple

import pyproj

from .core.base import Function
from .point_geo_projector import PointGeoProjector

class PolysGeoProjector(Function):
    def __init__(self) -> None:
        self.point_projector = PointGeoProjector()

    def __call__(self, *, polys: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
        new_polys = [[self.point_projector(point=p) for p in polygon] for polygon in polys]
        return new_polys
