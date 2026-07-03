from typing import Any, Dict, Literal

import numpy as np
import shapely

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter


class MassingSetback(SampleMetric):
    def __init__(self, *, massing_key: str,
                        site_key: str,
                        reduction: Literal["min", "max", "mean", "std"] = "mean") -> None:
        self.massing_key = massing_key
        self.site_key = site_key
        self.reduction = reduction
        self.polygons_converter = PolygonsToShapelyConverter()

    def _ground_footprint(self, *, building: Dict[str, Any]) -> shapely.Geometry:
        extrusions = building["massing"]
        if len(extrusions) == 0:
            return shapely.GeometryCollection()

        min_elevation = min([e["bottom_elevation"] for e in extrusions])
        footprints = []
        for extrusion in extrusions:
            if extrusion["bottom_elevation"] != min_elevation:
                continue
            polygons = [[(p[0], p[1]) for p in polygon] for polygon in extrusion["polygons"]]
            geometry = self.polygons_converter(polygons=polygons)
            geometry = shapely.make_valid(geometry, method='structure', keep_collapsed=False)
            footprints.append(geometry)
        return shapely.unary_union(footprints) if len(footprints) > 0 else shapely.GeometryCollection()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        site = [[(p[0], p[1]) for p in polygon] for polygon in sample[self.site_key]]
        site = self.polygons_converter(polygons=site)
        if site.is_empty:
            return 0.0

        values = []
        for building in sample[self.massing_key]:
            footprint = self._ground_footprint(building=building)
            if not footprint.is_empty:
                values.append(float(site.boundary.distance(footprint)))

        if len(values) == 0:
            return 0.0
        if self.reduction == "min":
            return float(min(values))
        if self.reduction == "max":
            return float(max(values))
        if self.reduction == "mean":
            return float(np.mean(values))
        if self.reduction == "std":
            return float(np.std(values))
        raise ValueError(f"Unknown reduction: {self.reduction}")
