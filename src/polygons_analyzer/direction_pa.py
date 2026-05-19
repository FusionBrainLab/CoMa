from typing import List, Any, Dict, Literal, Tuple
import math

import shapely
import numpy as np

from .polygons_analyzer import PolygonsAnalyzer
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class DirectionPA(PolygonsAnalyzer):
    def __init__(self) -> None:
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
    def __call__(self, *, polygons: List[List[Tuple[float, float]]]) -> float:
        polygon = self.polygons_to_shapely_converter(polygons=polygons)
        min_rect = polygon.minimum_rotated_rectangle
        coords = list(min_rect.exterior.coords)[:-1]
        
        distances = []
        for i in range(len(coords)):
            p1 = coords[i]
            p2 = coords[(i+1) % len(coords)]
            dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            distances.append(dist)
        
        longest_side_idx = np.argmax(distances)
        p1 = coords[longest_side_idx]
        p2 = coords[(longest_side_idx+1) % len(coords)]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        angle_deg = angle_deg % 180
        return angle_deg