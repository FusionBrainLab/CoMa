from typing import List, Any, Dict
import math

import pandas as pd
import numpy as np

from .dataset_processor import DatasetProcessor

class SiteMassingRotationAugmenter(DatasetProcessor):
    def __init__(self, *, site_col: str,
                        massing_col: str,
                        n_rotations: int) -> None:
        self.site_col = site_col
        self.massing_col = massing_col
        self.n_rotations = n_rotations
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        rotations = np.linspace(0, 2*math.pi, self.n_rotations, endpoint=False).tolist()
        new_dataset = {k: [] for k in pd_dataset.columns.to_list()}
        def rotate_point(base_point, target_point, angle):
            translated_x = target_point[0] - base_point[0]
            translated_y = target_point[1] - base_point[1]
            rotated_x = translated_x * math.cos(angle) - translated_y * math.sin(angle)
            rotated_y = translated_x * math.sin(angle) + translated_y * math.cos(angle)
            final_x = round(rotated_x + base_point[0], 2)
            final_y = round(rotated_y + base_point[1], 2)
            return (final_x, final_y)
        def augment_rotations(row):
            site = row[self.site_col]
            massing = row[self.massing_col]
            base_point = site[0][0]
            for r in rotations:
                new_site = [[rotate_point(base_point, p, r) for p in polygon] for polygon in site]
                new_massing = []
                for m in massing:
                    new_building = []
                    for e in m["massing"]:
                        new_e = {
                            "polygons": [[rotate_point(base_point, p, r) for p in polygon] for polygon in e["polygons"]],
                            "bottom_elevation":e["bottom_elevation"],
                            "top_elevation":e["top_elevation"]
                        }
                        new_building.append(new_e)
                    new_massing.append({
                        "id":m["id"],
                        "massing":new_building
                    })
                new_row = row.to_dict()
                new_row[self.site_col] = new_site
                new_row[self.massing_col] = new_massing
                for k, v in row.items():
                    new_dataset[k].append(v)
        pd_dataset.apply(lambda row: augment_rotations(row), axis=1)
        return new_dataset