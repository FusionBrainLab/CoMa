from typing import List, Any, Dict
import math
from copy import deepcopy

import pandas as pd
import numpy as np

from .dataset_processor import DatasetProcessor

class SiteMassingShiftAugmenter(DatasetProcessor):
    def __init__(self, *, site_col: str,
                        massing_col: str,
                        n_shifts: int,
                        shifts_range: List[float]) -> None:
        self.site_col = site_col
        self.massing_col = massing_col
        self.n_shifts = n_shifts
        self.shifts_range = shifts_range
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        shifts = np.linspace(self.shifts_range[0], self.shifts_range[1], self.n_shifts, endpoint=True).tolist()
        new_dataset = deepcopy(dataset)
        def augment(row):
            site = row[self.site_col]
            massing = row[self.massing_col]
            for s in shifts:
                for i in range(2):
                    for j in range(2):
                        new_site = [[(round(p[0] + s*(1**(i+1)), 2), round(p[1] + s*(1**(j+1)), 2)) for p in polygon] for polygon in site]
                        new_massing = []
                        for m in massing:
                            new_building = []
                            for e in m["massing"]:
                                new_e = {
                                    "polygons": [[(round(p[0] + s*(1**(i+1)), 2), round(p[1] + s*(1**(j+1)), 2)) for p in polygon] for polygon in e["polygons"]],
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
        pd_dataset.apply(lambda row: augment(row), axis=1)
        return new_dataset