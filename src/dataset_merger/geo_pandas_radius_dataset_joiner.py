from typing import Any, Dict, List

import pandas as pd
import geopandas as gpd

from .dataset_merger import DatasetMerger

class GeoPandasRadiusDatasetJoiner(DatasetMerger):
    def __init__(self, *, main_dataset_key: str,
                        join_dataset_key: str,
                        geo_col: str,
                        radius: float,
                        id_col: str,
                        joined_ids_col: str) -> None:
        self.main_dataset_key = main_dataset_key
        self.join_dataset_key = join_dataset_key
        self.geo_col = geo_col
        self.radius = radius
        self.id_col = id_col
        self.joined_ids_col = joined_ids_col
        
    def __call__(self, *, datasets: Dict[str, Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
        main_dataset = datasets[self.main_dataset_key]
        join_dataset = datasets[self.join_dataset_key]
        
        main_pd_dataset = pd.DataFrame(main_dataset)
        main_geo_dataset = gpd.GeoDataFrame(
            main_pd_dataset, 
            geometry=self.geo_col
        )
        join_pd_dataset = pd.DataFrame(join_dataset)
        join_geo_dataset = gpd.GeoDataFrame(
            join_pd_dataset, 
            geometry=self.geo_col
        )

        buffers = gpd.GeoDataFrame(
            geometry=join_geo_dataset.buffer(self.radius),
            index=join_geo_dataset.index
        )

        nearby_points = gpd.sjoin(
            buffers,          # Left: buffers around each point
            main_geo_dataset[[self.geo_col]], # Right: original points
            how='left',       # Keep all buffers
            predicate='intersects'  # Find points that intersect buffer (within radius)
        )

        results = {}
        for idx in tqdm(main_geo_dataset.index):
            # Get all points that intersect this point's buffer
            nearby_indices = nearby_points.loc[idx, 'index_right']
            
            # Handle different return types (single value vs array)
            if isinstance(nearby_indices, (int, np.integer)):
                if nearby_indices != idx:  # Exclude self
                    results[idx] = [nearby_indices]
            elif nearby_indices is not None:
                # Filter out self and convert to list
                filtered = [i for i in nearby_indices if i != idx]
                if filtered:
                    results[idx] = filtered
            else:
                results[idx] = []
        
        def get_ids(row):
            idx = row.name
            if idx not in results:
                return []
            context_idx = results[idx]
            context = main_geo_dataset.loc[context_idx][self.id_col].values.tolist()
            return context
        tqdm.pandas()
        main_geo_dataset[self.joined_ids_col] = main_geo_dataset.progress_apply(lambda row: get_ids(row), axis=1)

        new_dataset = main_geo_dataset.to_dict("list")
        return new_dataset