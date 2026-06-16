from typing import List, Any, Dict

import pandas as pd

from .dataset_processor import DatasetProcessor

class MassingAbsoluteCoordinatesProcessor(DatasetProcessor):
    def __init__(self, *, massing_col: str,
                        base_point_col: str) -> None:
        self.massing_col = massing_col
        self.base_point_col = base_point_col
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        def get_absolute_massing(row):
            try:
                massing = row[self.massing_col]
                base_point = row[self.base_point_col]
                new_massing = []
                for m in massing:
                    building = {"id": m["id"], "massing": []}
                    for extrusion in m["massing"]:
                        new_extrusion = {
                            "bottom_elevation": extrusion["bottom_elevation"],
                            "top_elevation": extrusion["top_elevation"],
                            "polygons": []
                        }
                        for polygon in extrusion["polygons"]:
                            new_extrusion["polygons"].append([
                                (round(p[0] + base_point[0], 2), round(p[1] + base_point[1], 2))
                                for p in polygon
                            ])
                        building["massing"].append(new_extrusion)
                    new_massing.append(building)
                return new_massing
            except Exception:
                return []
        pd_dataset[self.massing_col] = pd_dataset.apply(get_absolute_massing, axis=1)
        return pd_dataset.to_dict("list")