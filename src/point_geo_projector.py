from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple

import pyproj

from .core.base import Function

class PointGeoProjector(Function):
    def __call__(self, *, point: Tuple[float, float]) -> Tuple[float, float]:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        x, y = transformer.transform(*point)
        return (x, y)