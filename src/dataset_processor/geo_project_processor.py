from typing import List, Any, Dict

import pandas as pd
import geopandas as gpd

from .dataset_processor import DatasetProcessor

class GeoProjectProcessor(DatasetProcessor):
    def __init__(self, *, geo_col: str,
                        from_format: str,
                        to_format: str) -> None:
        self.geo_col = geo_col
        self.from_format = from_format
        self.to_format = to_format
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        from_gdf = gpd.GeoDataFrame(pd_dataset, geometry=self.geo_col, crs=self.from_format)
        to_gdf = from_gdf.to_crs(self.to_format)

        output_dataset = to_gdf.to_dict("list")
        return output_dataset