from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

import shapely
import trimesh

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class MeshSiteIoU(SampleMetric):
    def __init__(self, *, output_col: str) -> None:
        self.output_col = output_col
        self.polygons_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        site = [[(p[0], p[1]) for p in polygon] for polygon in sample["site_contour"]]
        site = self.polygons_converter(polygons=site)
        
        meshes = [trimesh.load(m["massing"]) for m in sample[self.output_col]]
        pred_mesh = meshes[0]
        for i in range(1, len(meshes)):
            try:
                pred_mesh = pred_mesh.union(meshes[i])
            except:
                pred_mesh = trimesh.util.concatenate([pred_mesh, meshes[i]])
        pred_bottom = trimesh.path.polygons.projected(pred_mesh, normal=[0,0,1])
        
        iou = shapely.area(site.intersection(pred_bottom))/shapely.area(site.union(pred_bottom))
        if str(iou) == "nan":
            iou = 0
        return iou