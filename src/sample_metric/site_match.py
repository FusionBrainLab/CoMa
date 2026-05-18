from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

import shapely
import trimesh

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class SiteMatch(SampleMetric):
    def __init__(self, *, pred_col: str,
                        pred_type: Literal["polygon", "mesh"]) -> None:
        self.pred_col = pred_col
        self.pred_type = pred_type
        self.polygons_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        site = [[(p[0], p[1]) for p in polygon] for polygon in sample["site_contour"]]
        site = self.polygons_converter(polygons=site)
        
        if self.pred_type == "polygon":
            pred_massings = sample[self.pred_col]
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
        elif self.pred_type == "mesh":
            meshes = [trimesh.load(m["massing"]) for m in sample[self.pred_col]]
            pred_mesh = meshes[0]
            for i in range(1, len(meshes)):
                try:
                    pred_mesh = pred_mesh.union(meshes[i])
                except:
                    pred_mesh = trimesh.util.concatenate([pred_mesh, meshes[i]])
            pred_bottom = trimesh.path.polygons.projected(pred_mesh, normal=[0,0,1])
        
        site_match = shapely.area(site.intersection(pred_bottom))/shapely.area(pred_bottom)
        if str(site_match) == "nan":
            site_match = 0
        return site_match