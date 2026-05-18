from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
import json

from .core.base import Function

class MDPolygonsConverter(Function):
    def __call__(self, *, md_geo_shape: str) -> List[List[Tuple[float, float]]]:
        raw_data = json.loads(md_geo_shape)
        raw_polygons = raw_data["coordinates"]
        if raw_data["type"] == "Polygon":
            raw_polygons = [raw_polygons]
        polygons = []
        for p_list in raw_polygons:
            for p in p_list:
                p = [(pol_point[0], pol_point[1]) for pol_point in p]
                polygons.append(p)
        return polygons