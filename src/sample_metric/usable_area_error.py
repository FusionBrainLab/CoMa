from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

import shapely

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class UsableAreaError(SampleMetric):
    def __init__(self, *, output_col: str,
                        floor_height_key: str,
                        area_key: str) -> None:
        self.output_col = output_col
        self.floor_height_key = floor_height_key
        self.area_key = area_key
        self.polygons_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        total_gt_requirements = sample["requirements"]
        error = 0
        computed = 0
        for gt_requirements in total_gt_requirements:
            gt_floor_height = gt_requirements[self.floor_height_key]
            gt_area = gt_requirements[self.area_key]

            pred_massing = [m for m in sample[self.output_col] if str(m["id"]) == str(gt_requirements["id"])]
            if len(pred_massing) == 0:
                continue
            pred_massing = pred_massing[0]["massing"]
            min_elevation = min([e["bottom_elevation"] for e in pred_massing])
            max_elevation = max([e["top_elevation"] for e in pred_massing])
            pred_height = max_elevation - min_elevation

            pred_n_floors = math.ceil(pred_height / gt_floor_height)

            area = 0
            heights = [min_elevation] + [gt_floor_height * (i + 1) for i in range(pred_n_floors - 1)] if pred_n_floors > 1 else [min_elevation]
            for local_height in heights:
                local_footprints = [e["polygons"] for e in pred_massing if local_height >= e["bottom_elevation"] and local_height < e["top_elevation"]]
                local_footprints = [[[(p[0], p[1]) for p in polygon] for polygon in foot] for foot in local_footprints]
                local_footprints = [self.polygons_converter(polygons=p) for p in local_footprints]
                if len(local_footprints) == 0:
                    continue
                local_floor = local_footprints[0]
                for i in range(1, len(local_footprints)):
                    local_floor = local_floor.union(local_footprints[i])
                local_area = shapely.area(local_floor)
                area += local_area

            local_error = abs(gt_area - area)/gt_area
            error += local_error
            computed += 1

        if computed == 0:
            raise
        
        error = error/computed
        return error