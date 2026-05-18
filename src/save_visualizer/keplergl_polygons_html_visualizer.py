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

class KeplerGLPolygonsHTMLVisualizer(SaveVisualizer):
    def __init__(self, *, path_template: str,
                        dataset_key: str,
                        polygons_col: str) -> None:
        self.path_template = path_template
        self.dataset_key = dataset_key
        self.polygons_col = polygons_col
        self.string_formatter = StringFormatter()

    def __call__(self, *, data: Dict[str, Any]) -> str:
        dataset = data[self.dataset_key]
        dataset = pd.DataFrame(dataset)

        geo_dataset = gpd.GeoDataFrame(
            dataset, 
            geometry=self.polygons_col
        )

        map_1 = KeplerGl(height=600, 
                        data={"data": geo_dataset},
                        config={
                            "version": "v1",
                            "config": {
                                "visState": {
                                    "filters": [],
                                    "layers": [{
                                        "type": "geojson",
                                        "config": {
                                            "dataId": "data",
                                            "label": "data",
                                            "color": [18, 147, 154],
                                            "columns": {"geojson": self.polygons_col},
                                            "isVisible": True,
                                            "visConfig": {
                                                "opacity": 0.8,
                                                "strokeColor": [255, 255, 255],
                                                "thickness": 0.1
                                            }
                                        }
                                    }]
                                }
                            }
                        })

        path = self.string_formatter(string=self.path_template, args=data)
        map_1.save_to_html(file_name=path)
        return path
        