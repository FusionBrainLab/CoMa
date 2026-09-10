from typing import List, Dict, Any
import io
import math

import kaleido
from PIL import Image
import plotly.graph_objects as go
import pyproj

from .image_data_visualizer import ImageDataVisualizer

class RegionMetricMapVisualizer(ImageDataVisualizer):
    def __init__(self, *, dataset_key: str,
                        polygon_col: str,
                        base_point_col: str,
                        metric_col: str,
                        color_scale: str,
                        colorbar_title: str,
                        value_range: List[float],
                        window_size: List[int],
                        map_type: str,
                        from_crs: str,
                        zoom_level: int,
                        render_timeout: int) -> None:
        self.dataset_key = dataset_key
        self.polygon_col = polygon_col
        self.base_point_col = base_point_col
        self.metric_col = metric_col
        self.color_scale = color_scale
        self.colorbar_title = colorbar_title
        self.value_range = value_range
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
        dataset = data[self.dataset_key]
        features = []
        values = []
        total_lons = []
        total_lats = []
        for i, (region, base_point, value) in enumerate(zip(
            dataset[self.polygon_col],
            dataset[self.base_point_col],
            dataset[self.metric_col]
        )):
            geo_polygons = []
            for polygon in region:
                geo_polygon = [
                    self.transformer.transform(
                        point[0] + base_point[0],
                        point[1] + base_point[1]
                    )
                    for point in polygon
                ]
                if len(geo_polygon) == 0:
                    continue
                geo_polygon.append(geo_polygon[0])
                total_lons.extend(point[0] for point in geo_polygon)
                total_lats.extend(point[1] for point in geo_polygon)
                geo_polygons.append([geo_polygon])
            if len(geo_polygons) == 0:
                continue
            features.append({
                "type": "Feature",
                "properties": {"index": i},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": geo_polygons
                }
            })
            values.append(value)

        min_lon, max_lon = min(total_lons), max(total_lons)
        min_lat, max_lat = min(total_lats), max(total_lats)
        max_range = max(max_lon - min_lon, max_lat - min_lat, 1e-12)
        zoom_level = max(
            1,
            min(20, self.zoom_level - math.log(max_range * 100))
        )

        map_fig = go.Figure(
            go.Choroplethmapbox(
                geojson={
                    "type": "FeatureCollection",
                    "features": features
                },
                locations=[
                    feature["properties"]["index"]
                    for feature in features
                ],
                z=values,
                featureidkey="properties.index",
                colorscale=self.color_scale,
                zmin=self.value_range[0],
                zmax=self.value_range[1],
                marker_opacity=0.8,
                marker_line_width=0.5,
                colorbar={
                    "title": {
                        "text": self.colorbar_title,
                        "font": {"size": 32}
                    },
                    "tickfont": {"size": 28},
                    "orientation": "h",
                    "len": 0.9,
                    "thickness": 50,
                    "x": 0.5,
                    "xanchor": "center",
                    "y": -0.12,
                    "yanchor": "top"
                }
            )
        )
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
            margin={"l": 0, "r": 0, "t": 0, "b": 180}
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
