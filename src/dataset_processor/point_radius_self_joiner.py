from typing import List, Any, Dict

import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import pandas as pd
from tqdm import tqdm

from .dataset_processor import DatasetProcessor

class PointRadiusSelfJoiner(DatasetProcessor):
    def __init__(self, *, point_col: str,
                        radius: float,
                        id_col: str,
                        joined_ids_col: str) -> None:
        self.point_col = point_col
        self.radius = radius
        self.id_col = id_col
        self.joined_ids_col = joined_ids_col
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        geo_dataset = gpd.GeoDataFrame(
            pd_dataset, 
            geometry=self.point_col
        )

        buffers = gpd.GeoDataFrame(
            geometry=geo_dataset.buffer(self.radius),
            index=geo_dataset.index
        )

        nearby_points = gpd.sjoin(
            buffers,          # Left: buffers around each point
            geo_dataset[[self.point_col]], # Right: original points
            how='left',       # Keep all buffers
            predicate='intersects'  # Find points that intersect buffer (within radius)
        )

        results = {}
        for idx in tqdm(geo_dataset.index):
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
            context = geo_dataset.loc[context_idx][self.id_col].values.tolist()
            return context
        tqdm.pandas()
        geo_dataset[self.joined_ids_col] = geo_dataset.progress_apply(lambda row: get_ids(row), axis=1)

        """geo_dataset[self.joined_ids_col] = geo_dataset.index.map(
            lambda idx: [str(geo_dataset.loc[neighbor_idx, self.id_col]) for neighbor_idx in results.get(idx, [])]
        )"""
        new_dataset = geo_dataset.to_dict("list")
        return new_dataset