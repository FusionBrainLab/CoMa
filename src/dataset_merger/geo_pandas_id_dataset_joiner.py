from typing import Any, Dict, List

import pandas as pd
import geopandas as gpd

from .dataset_merger import DatasetMerger

class GeoPandasIdDatasetJoiner(DatasetMerger):
    def __init__(self, *, join_dataset_key: str,
                        join_dataset_geo_col: str,
                        join_dataset_id_col: str,
                        main_dataset_key: str,
                        main_dataset_geo_col: str,
                        predicate: str,
                        output_col: str) -> None:
        self.join_dataset_key = join_dataset_key
        self.join_dataset_geo_col = join_dataset_geo_col
        self.join_dataset_id_col = join_dataset_id_col
        self.main_dataset_key = main_dataset_key
        self.main_dataset_geo_col = main_dataset_geo_col
        self.predicate = predicate
        self.output_col = output_col
        
    def __call__(self, *, datasets: Dict[str, Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
        join_dataset = datasets[self.join_dataset_key]
        main_dataset = datasets[self.main_dataset_key]
        
        main_pd_dataset = pd.DataFrame(main_dataset)
        main_geo_dataset = gpd.GeoDataFrame(
            main_pd_dataset, 
            geometry=self.main_dataset_geo_col
        )
        join_geo_dataset = gpd.GeoDataFrame(
            pd.DataFrame(join_dataset), 
            geometry=self.join_dataset_geo_col
        )
        new_dataset = gpd.sjoin(
            join_geo_dataset, 
            main_geo_dataset, 
            how='inner', 
            predicate=self.predicate
        )
        join_dataset_ids = new_dataset.groupby(f'index_right')[f"{self.join_dataset_id_col}_left"].apply(list).to_dict()
        main_pd_dataset[self.output_col] = [[] for _ in range(len(main_pd_dataset))]
        for self_idx, join_ids in join_dataset_ids.items():
            main_pd_dataset.at[self_idx, self.output_col] = join_ids
        
        dataset = main_pd_dataset.to_dict("list")
        return dataset