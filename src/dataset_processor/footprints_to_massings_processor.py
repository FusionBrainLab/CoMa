from typing import List, Any, Dict

import pandas as pd

from .dataset_processor import DatasetProcessor
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter

class FootprintsToMassingsProcessor(DatasetProcessor):
    def __init__(self, *, polygons_col: str,
                        bottom_elevation_col: str,
                        top_elevation_col: str,
                        building_id_col: str,
                        output_col: str) -> None:
        self.polygons_col = polygons_col
        self.bottom_elevation_col = bottom_elevation_col
        self.top_elevation_col = top_elevation_col
        self.building_id_col = building_id_col
        self.output_col = output_col
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        pd_buildings = pd_dataset.groupby(self.building_id_col).agg(list)

        def get_massing(row):
            bottom_elevations = row[self.bottom_elevation_col]
            top_elevations = row[self.top_elevation_col]
            min_elevation = min(bottom_elevations)
            bottom_elevations = [e - min_elevation for e in bottom_elevations]
            top_elevations = [e - min_elevation for e in top_elevations]
            polygons = row[self.polygons_col]
            massing = []
            for plist, bottom, top in zip(polygons, bottom_elevations, top_elevations):
                extrusion = {
                    "polygons":self.shapely_to_polygons_converter(polygons=plist),
                    "bottom_elevation":bottom,
                    "top_elevation":top
                }
                massing.append(extrusion)
            return massing
        pd_buildings[self.output_col] = pd_buildings.progress_apply(lambda row: get_massing(row), axis=1)

        output_dataset = pd_buildings.to_dict("list")
        return output_dataset