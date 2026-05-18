from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

import trimesh
import shapely
from trimesh import transformations
import numpy as np

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter
from ..mesh_compiler import MassingMeshCompiler

def get_polygon_elongation_direction(polygon):
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

class ExtrusionsIoU(SampleMetric):
    def __init__(self, *, gt_col: str,
                        pred_col: str) -> None:
        self.gt_col = gt_col
        self.pred_col = pred_col
        self.mesh_creator = MassingMeshCompiler()
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        def get_footprint(massing):
            polygons = [[[(point[0], point[1]) for point in polygon] for polygon in e["polygons"]] for e in massing]
            footprint = shapely.unary_union([self.polygons_to_shapely_converter(polygons=p) for p in polygons])
            return footprint

        total_gt_massings = sample[self.gt_col]
        total_pred_massings = sample[self.pred_col]

        value = 0
        computed = 0
        for gt in total_gt_massings:
            pred = [m for m in total_pred_massings if m["id"] == gt["id"]]
            if len(pred) == 0:
                continue
            pred = pred[0]
            
            gt_mesh = self.mesh_creator(obj=[gt])
            pred_mesh = self.mesh_creator(obj=[pred])
            gt_footprint = get_footprint(gt["massing"])
            pred_footprint = get_footprint(pred["massing"])

            #Get translation
            gt_centroid = shapely.centroid(gt_footprint)
            pred_centroid = shapely.centroid(pred_footprint)
            translation = np.array([gt_centroid.x - pred_centroid.x, gt_centroid.y - pred_centroid.y, 0])
            pred_mesh.apply_translation(translation)

            #Get rotation
            gt_dir = get_polygon_elongation_direction(gt_footprint)
            pred_dir = get_polygon_elongation_direction(pred_footprint)
            angle_diff = np.radians(gt_dir - pred_dir)
            rotation_matrix = transformations.rotation_matrix(angle_diff, [0, 0, 1], pred_mesh.centroid)
            pred_mesh.apply_transform(rotation_matrix)

            gt_emeshes = [self.mesh_creator(obj=[{"id":"0", "massing":[e]}]) for e in gt["massing"]]
            for e in pred["massing"]:
                emesh = self.mesh_creator(obj=[{"id":"0", "massing":[e]}])
                max_iou = 0
                for gt_emesh in gt_emeshes:
                    intersection = gt_emesh.intersection(emesh)
                    union = gt_emesh.union(emesh)
                    iou = intersection.volume/union.volume
                    if iou > max_iou:
                        max_iou = iou
                value += max_iou
                computed += 1

        result = value / computed if computed > 0 else 0
        return result