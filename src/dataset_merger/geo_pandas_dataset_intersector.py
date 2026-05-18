from typing import Any, Dict, List, Literal

import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import shapely

from .dataset_merger import DatasetMerger

class GeoPandasDatasetIntersector(DatasetMerger):
    def __init__(self, *, intersect_dataset_key: str,
                        intersect_dataset_geo_col: str,
                        intersect_dataset_id_col: str,
                        main_dataset_key: str,
                        main_dataset_geo_col: str,
                        main_dataset_id_col: str,
                        drop_self_intersections: bool,
                        output_properties: List[Literal["id", "geometry", "area"]],
                        output_col: str) -> None:
        self.intersect_dataset_key = intersect_dataset_key
        self.intersect_dataset_geo_col = intersect_dataset_geo_col
        self.intersect_dataset_id_col = intersect_dataset_id_col
        self.main_dataset_key = main_dataset_key
        self.main_dataset_geo_col = main_dataset_geo_col
        self.main_dataset_id_col = main_dataset_id_col
        self.drop_self_intersections = drop_self_intersections
        self.output_properties = output_properties
        self.output_col = output_col
        
    def __call__(self, *, datasets: Dict[str, Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
        main_dataset = datasets[self.main_dataset_key]
        intersect_dataset = datasets[self.intersect_dataset_key]
        
        main_pd_dataset = pd.DataFrame(main_dataset)
        main_id_col = f"main_{self.main_dataset_id_col}"
        main_pd_dataset[main_id_col] = main_pd_dataset[self.main_dataset_id_col]
        main_pd_dataset["geometry"] = main_pd_dataset[self.main_dataset_geo_col]
        main_geo_dataset = gpd.GeoDataFrame(
            main_pd_dataset, 
            geometry="geometry"
        )
        intersect_pd_dataset = pd.DataFrame(intersect_dataset)
        intersect_id_col = f"intersect_{self.intersect_dataset_id_col}"
        intersect_pd_dataset[intersect_id_col] = intersect_pd_dataset[self.intersect_dataset_id_col]
        intersect_pd_dataset["geometry"] = intersect_pd_dataset[self.intersect_dataset_geo_col]
        intersect_geo_dataset = gpd.GeoDataFrame(
            intersect_pd_dataset, 
            geometry="geometry"
        )
        intersection_dataset = main_geo_dataset.overlay(intersect_geo_dataset, how='intersection', keep_geom_type=False)
        intersection_dataset[f"{main_id_col}_buffer"] = intersection_dataset[main_id_col]
        intersection_dataset = intersection_dataset.groupby(f"{main_id_col}_buffer").agg(list)
        intersection_dataset[main_id_col] = intersection_dataset.apply(lambda row: row[main_id_col][0], axis=1)
        intersection_dataset[f"{main_id_col}_buffer"] = intersection_dataset[main_id_col]
        intersection_dataset = intersection_dataset.set_index(f"{main_id_col}_buffer")

        tqdm.pandas()
        def get_intersections(row):
            pd_intersections = intersection_dataset[["geometry", intersect_id_col]].loc[row[main_id_col]].to_dict() if row[main_id_col] in intersection_dataset[main_id_col].values else None
            if pd_intersections == None:
                return []
            intersections = [{"id": pd_intersections[intersect_id_col][i], "geometry": pd_intersections["geometry"][i]} for i in range(len(pd_intersections["geometry"]))]
            new_intersections = []
            for spec in intersections:
                new_spec = {k: v for k, v in spec.items() if k in self.output_properties}
                if "area" in self.output_properties:
                    area = shapely.area(spec["geometry"])
                    new_spec["area"] = area if area else 0
                new_intersections.append(new_spec)
            intersections = new_intersections

            if self.drop_self_intersections:
                intersections = [spec for spec in intersections if spec["id"] != row["id"]]
            return intersections
        main_pd_dataset[self.output_col] = main_pd_dataset.progress_apply(lambda row: get_intersections(row), axis=1)
        main_pd_dataset = main_pd_dataset.drop(columns=[main_id_col, "geometry"])
        dataset = main_pd_dataset.to_dict("list")
        return dataset