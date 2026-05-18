from typing import List, Dict, Any
import re
import json

import trimesh
import pandas as pd
import numpy as np
import math
from keplergl import KeplerGl
import geopandas as gpd

from .save_visualizer import SaveVisualizer
from ..string_formatter import StringFormatter

class KeplerGLMultiPolygonsHTMLVisualizer(SaveVisualizer):
    def __init__(self, *, path_template: str,
                        dataset_key: str,
                        polygon_cols: List[str],
                        id_col: str) -> None:
        self.path_template = path_template
        self.dataset_key = dataset_key
        self.polygon_cols = polygon_cols
        self.id_col = id_col
        self.string_formatter = StringFormatter()

    def __call__(self, *, data: Dict[str, Any]) -> str:
        dataset = data[self.dataset_key]
        dataset = pd.DataFrame(dataset)

        # Define colors for different polygon layers
        colors = [
            [18, 147, 154],  # Teal
            [255, 99, 71],   # Tomato
            [75, 192, 192],  # Mint
            [255, 206, 86],  # Gold
            [153, 102, 255], # Purple
            [255, 159, 64],  # Orange
            [201, 203, 207], # Gray
            [255, 99, 132]   # Pink
        ]

        # Create a separate dataset for each polygon column
        datasets = {}
        layers = []

        for i, poly_col in enumerate(self.polygon_cols):
            # Create a unique dataset name for this polygon column
            dataset_name = f"data_{poly_col}"
            
            # Create a GeoDataFrame with this polygon column renamed to 'geometry'
            # Also include id column for reference
            cols_to_include = [self.id_col] + [col for col in dataset.columns if col != poly_col and col != self.id_col]
            
            geo_df = gpd.GeoDataFrame(
                dataset[cols_to_include].copy(),
                geometry=dataset[poly_col]  # This will be renamed to 'geometry' automatically
            )
            
            # Add to datasets dictionary
            datasets[dataset_name] = geo_df
            
            # Create layer for this dataset
            color = colors[i % len(colors)]
            
            layer = {
                "type": "geojson",
                "config": {
                    "dataId": dataset_name,
                    "label": f"{poly_col}",
                    "color": color,
                    "columns": {"geojson": "geometry"},  # Use 'geometry' column
                    "isVisible": True,
                    "visConfig": {
                        "opacity": 0.8,
                        "strokeColor": [255, 255, 255],
                        "thickness": 0.1,
                        "strokeOpacity": 0.8
                    }
                }
            }
            
            # Add tooltip configuration if id column exists
            if 'id' in dataset.columns:
                layer['config']['tooltips'] = ['id']
            
            layers.append(layer)

        # Create the Kepler.gl map with multiple datasets
        map_1 = KeplerGl(
            height=600, 
            data=datasets,  # Pass dictionary of datasets
            config={
                "version": "v1",
                "config": {
                    "visState": {
                        "filters": [],
                        "layers": layers
                    }
                }
            }
        )

        # Save to HTML
        path = self.string_formatter(string=self.path_template, args=data)
        map_1.save_to_html(file_name=path)
        return path