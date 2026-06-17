from typing import Any, Dict
import math

import shapely

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter


class MassingFloorAreaRatio(SampleMetric):
    def __init__(self, *, massing_key: str,
                        site_key: str,
                        floor_height: float = 3.0) -> None:
        self.massing_key = massing_key
        self.site_key = site_key
        self.floor_height = floor_height
        self.polygons_converter = PolygonsToShapelyConverter()

    def _extrusion_footprint(self, extrusion: Dict[str, Any]) -> shapely.Geometry:
        polygons = [[(p[0], p[1]) for p in polygon] for polygon in extrusion["polygons"]]
        geometry = self.polygons_converter(polygons=polygons)
        geometry = shapely.make_valid(geometry, method='structure', keep_collapsed=False)
        return geometry

    def _floor_area(self, *, building: Dict[str, Any]) -> float:
        extrusions = building["massing"]
        if len(extrusions) == 0 or self.floor_height <= 0:
            return 0.0

        min_elevation = min([e["bottom_elevation"] for e in extrusions])
        max_elevation = max([e["top_elevation"] for e in extrusions])
        n_floors = math.ceil((max_elevation - min_elevation) / self.floor_height)

        area = 0.0
        heights = [min_elevation + self.floor_height * i for i in range(n_floors)]
        for local_height in heights:
            local_footprints = [
                self._extrusion_footprint(e)
                for e in extrusions
                if local_height >= e["bottom_elevation"] and local_height < e["top_elevation"]
            ]
            if len(local_footprints) == 0:
                continue
            local_floor = shapely.unary_union(local_footprints)
            area += shapely.area(local_floor)
        return area

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        site = [[(p[0], p[1]) for p in polygon] for polygon in sample[self.site_key]]
        site = self.polygons_converter(polygons=site)
        site_area = shapely.area(site)
        if site_area <= 0:
            return 0.0

        floor_area = sum([self._floor_area(building=m) for m in sample[self.massing_key]])
        return floor_area / site_area
