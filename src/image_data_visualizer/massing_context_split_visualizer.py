from typing import List, Dict, Any
import re
import json

import pyvista as pv
import trimesh
from PIL import Image
import pandas as pd
from pandarallel import pandarallel
import numpy as np
import math
import pyproj

from .image_data_visualizer import ImageDataVisualizer
from ..dataset_creator import DatasetCreator

class MassingContextSplitVisualizer(ImageDataVisualizer):
    def __init__(self, *, train_dataset_loader: DatasetCreator,
                        test_dataset_loader: DatasetCreator,
                        id_col: str,
                        train_context_ids_col: str,
                        test_context_ids_col: str,
                        site_contour_col: str,
                        base_point_col: str,
                        category_colors: Dict[int, str],
                        window_size: List[int],
                        map_type: str,
                        from_crs: str,
                        zoom_level: int,
                        render_timeout: int) -> None:
        train_dataset = pd.DataFrame(train_dataset_loader())
        test_dataset = pd.DataFrame(test_dataset_loader())
        site_dataset = pd.concat([train_dataset, test_dataset]).set_index(id_col)

        train_gt_ids = set(train_dataset[id_col])
        train_context_ids = {
            site_id
            for context_ids in train_dataset[train_context_ids_col]
            for site_id in context_ids
        }
        test_context_ids = {
            site_id
            for context_ids in test_dataset[test_context_ids_col]
            for site_id in context_ids
        }

        self.site_regions = {}
        self.site_categories = {}
        for site_id in test_context_ids:
            if site_id not in site_dataset.index:
                continue
            site = site_dataset.loc[site_id]
            base_point = site[base_point_col]
            site_contour = site[site_contour_col]
            if isinstance(site_contour, str):
                site_contour = json.loads(site_contour)
            self.site_regions[site_id] = [
                [
                    (point[0] + base_point[0], point[1] + base_point[1])
                    for point in polygon
                ]
                for polygon in site_contour
            ]
            self.site_categories[site_id] = (
                1
                + (site_id in train_context_ids)
                + 2 * (site_id in train_gt_ids)
            )

        self.category_colors = category_colors
        self.window_size = window_size
        self.map_type = map_type
        self.zoom_level = zoom_level
        self.render_timeout = render_timeout
        self.transformer = pyproj.Transformer.from_crs(
            from_crs,
            "EPSG:4326",
            always_xy=True
        )

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        import kaleido

        import io
        import plotly.graph_objects as go

        map_fig = go.Figure()
        total_lons = []
        total_lats = []
        for category in range(1, 5):
            category_lons = []
            category_lats = []
            for site_id, site_category in self.site_categories.items():
                if site_category != category:
                    continue
                for polygon in self.site_regions[site_id]:
                    if len(polygon) == 0:
                        continue
                    geo_polygon = [
                        self.transformer.transform(point[0], point[1])
                        for point in polygon
                    ]
                    lons = [point[0] for point in geo_polygon] + [geo_polygon[0][0]]
                    lats = [point[1] for point in geo_polygon] + [geo_polygon[0][1]]
                    total_lons.extend(lons)
                    total_lats.extend(lats)
                    category_lons.extend(lons + [None])
                    category_lats.extend(lats + [None])
            if len(category_lons) > 0:
                map_fig.add_trace(
                    go.Scattermapbox(
                        mode="lines",
                        lon=category_lons,
                        lat=category_lats,
                        fill="toself",
                        fillcolor=self.category_colors[category],
                        line=dict(width=1, color="white"),
                        name=[
                            "Test context only",
                            "Train and test context",
                            "Test context and train ground truth",
                            "Train/test context and train ground truth"
                        ][category - 1]
                    )
                )

        min_lon, max_lon = min(total_lons), max(total_lons)
        min_lat, max_lat = min(total_lats), max(total_lats)
        max_range = max(max_lon - min_lon, max_lat - min_lat, 1e-12)
        zoom_level = max(1, min(20, self.zoom_level - math.log(max_range * 100)))

        map_fig.update_layout(
            mapbox={
                "style": self.map_type,
                "center": {
                    "lon": sum(total_lons) / len(total_lons),
                    "lat": sum(total_lats) / len(total_lats)
                },
                "zoom": zoom_level
            },
            width=self.window_size[0],
            height=self.window_size[1],
            margin={"l": 0, "r": 0, "t": 0, "b": 0}
        )

        image = kaleido.calc_fig_sync(
            map_fig,
            opts={
                "format": "png",
                "width": self.window_size[0],
                "height": self.window_size[1]
            },
            kopts={"timeout": self.render_timeout}
        )
        return Image.open(io.BytesIO(image))