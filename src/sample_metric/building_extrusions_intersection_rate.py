from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

import shapely
from shapely import unary_union
from shapely.geometry import MultiPolygon, Polygon

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class BuildingExtrusionsIntersectionRate(SampleMetric):
    def __init__(self, *, building_key: str) -> None:
        self.building_key = building_key
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        building = sample[self.building_key]
        
        counter = 0
        value = 0
        for i in range(len(building["massing"])):
            for j in range(i + 1, len(building["massing"])):
                e1 = building["massing"][i]
                e2 = building["massing"][j]
                p1 = self.polygons_to_shapely_converter(polygons=[[(point[0], point[1]) for point in polygon] for polygon in e1["polygons"]])
                p2 = self.polygons_to_shapely_converter(polygons=[[(point[0], point[1]) for point in polygon] for polygon in e2["polygons"]])

                intersection = p1.intersection(p2)
                difference = p1.difference(p2)

                elevation_case = None
                if (e1["bottom_elevation"] >= e2["bottom_elevation"] and 
                    e1["bottom_elevation"] < e2["top_elevation"] and 
                    e1["top_elevation"] > e2["top_elevation"]):
                    elevation_case = 0
                elif (e1["bottom_elevation"] < e2["bottom_elevation"] and 
                    e1["top_elevation"] > e2["bottom_elevation"] and 
                    e1["top_elevation"] <= e2["top_elevation"]):
                    elevation_case = 1
                elif (e2["bottom_elevation"] > e1["bottom_elevation"] and 
                    e2["top_elevation"] < e1["top_elevation"]):
                    elevation_case = 2
                elif (e1["bottom_elevation"] > e2["bottom_elevation"] and 
                    e1["top_elevation"] < e2["top_elevation"]):
                    elevation_case = 3
                elif (e1["bottom_elevation"] > e2["top_elevation"] or 
                    e1["top_elevation"] < e2["bottom_elevation"]):
                    elevation_case = 4
                
                polygon_case = None
                if abs(shapely.area(intersection) - shapely.area(p1)) < 1e-6:
                    polygon_case = 0
                elif shapely.area(intersection) == 0:
                    polygon_case = 3
                else:
                    if type(difference) == Polygon:
                        polygon_case = 1
                    elif type(difference) == MultiPolygon:
                        polygon_case = 2
                    else:
                        continue
                
                if polygon_case != 3 and elevation_case != 4:
                    value += 1
                counter += 1
                
        value = value/counter if counter > 0 else 0
        return value