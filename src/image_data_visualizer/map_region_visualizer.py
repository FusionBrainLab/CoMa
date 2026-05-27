from typing import List, Dict, Any
import io
import math

import plotly.graph_objects as go
import pyproj
from PIL import Image

from .image_data_visualizer import ImageDataVisualizer

class MapRegionVisualizer(ImageDataVisualizer):
    def __init__(self, *, region_key: str,
                        fill_color: str,
                        border_color: str,
                        window_size: List[int],
                        map_type: str,
                        from_crs: str,
                        zoom_level: int) -> None:
        self.region_key = region_key
        self.fill_color = fill_color
        self.border_color = border_color
        self.window_size = window_size
        self.map_type = map_type
        self.zoom_level = zoom_level
        self.transformer = pyproj.Transformer.from_crs(
            from_crs,
            "EPSG:4326",
            always_xy=True
        )

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        region = data[self.region_key]
        geo_region = []
        for polygon in region:
            geo_polygon = []
            for point in polygon:
                lon, lat = self.transformer.transform(point[0], point[1])
                geo_polygon.append((lon, lat))
            geo_region.append(geo_polygon)

        map_fig = go.Figure()
        total_lons = []
        total_lats = []
        for polygon in geo_region:
            if len(polygon) == 0:
                continue

            lons = [p[0] for p in polygon] + [polygon[0][0]]
            lats = [p[1] for p in polygon] + [polygon[0][1]]
            total_lons.extend(lons)
            total_lats.extend(lats)
            map_fig.add_trace(
                go.Scattermapbox(
                    mode="lines",
                    lon=lons,
                    lat=lats,
                    fill="toself",
                    fillcolor=self.fill_color,
                    line=dict(width=2, color=self.border_color)
                )
            )

        min_lon, max_lon = min(total_lons), max(total_lons)
        min_lat, max_lat = min(total_lats), max(total_lats)
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        max_range = max(lon_range, lat_range, 1e-12)

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

        image = map_fig.to_image(
            format="png",
            width=self.window_size[0],
            height=self.window_size[1]
        )
        return Image.open(io.BytesIO(image))