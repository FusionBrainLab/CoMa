from abc import ABC, abstractmethod
from typing import List, Any, Dict
import math
import json

import trimesh
import numpy as np

from .sample_metric import SampleMetric

class MeshUsableAreaError(SampleMetric):
    def __init__(self, *, output_col: str) -> None:
        self.output_col = output_col

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        total_gt_requirements = sample["requirements"]
        error = 0
        computed = 0
        for gt_requirements in total_gt_requirements:
            gt_floor_height = gt_requirements["floor_height"]
            gt_area = gt_requirements["usable_area"]

            pred_massing = [m for m in sample[self.output_col] if m["id"] == str(gt_requirements["id"])]
            if len(pred_massing) == 0:
                continue
            pred_massing = pred_massing[0]["massing"]
            pred_mesh = trimesh.load(pred_massing)

            massing_vertices = pred_mesh.vertices
            z_min, z_max = np.min(massing_vertices[:, 2]), np.max(massing_vertices[:, 2])
            pred_height = z_max - z_min

            pred_n_floors = math.ceil(pred_height / gt_floor_height)

            area = pred_mesh.volume / pred_n_floors / gt_floor_height

            local_error = abs(gt_area - area)/gt_area
            error += local_error
            computed += 1

        if computed == 0:
            raise
        
        error = error/computed
        return error