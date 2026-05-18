from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

import shapely

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class SiteIoU(SampleMetric):
    def __init__(self, *, output_col: str) -> None:
        self.output_col = output_col
        self.polygons_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        site = [[(p[0], p[1]) for p in polygon] for polygon in sample["site_contour"]]
        site = self.polygons_converter(polygons=site)
        
        pred_massings = sample[self.output_col]
        pred_bottom_elevations = [min(e["bottom_elevation"] for e in m["massing"]) for m in pred_massings]
        def collect_bottoms(massing, elevation):
            bottoms = [e["polygons"] for e in massing if e["bottom_elevation"] == elevation]
            bottoms = [[[(p[0], p[1]) for p in polygon] for polygon in foot] for foot in bottoms]
            bottom = []
            for b in bottoms:
                bottom.extend(b)
            return bottom
        pred_bottoms = [collect_bottoms(pred_massings[i]["massing"], pred_bottom_elevations[i]) for i in range(len(pred_massings))]
        pred_bottoms = [self.polygons_converter(polygons=p) for p in pred_bottoms]
        pred_bottom = pred_bottoms[0]
        for i in range(1, len(pred_bottoms)):
            pred_bottom = pred_bottom.union(pred_bottoms[i])
        
        iou = shapely.area(site.intersection(pred_bottom))/shapely.area(site.union(pred_bottom))
        if str(iou) == "nan":
            iou = 0
        return iou