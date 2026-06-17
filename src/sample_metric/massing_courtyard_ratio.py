from typing import Any, Dict

import numpy as np
import shapely

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter


class MassingCourtyardRatio(SampleMetric):
    def __init__(self, *, massing_key: str) -> None:
        self.massing_key = massing_key
        self.polygons_converter = PolygonsToShapelyConverter()

    def _footprint(self, *, massing) -> shapely.Geometry:
        footprints = []
        for building in massing:
            for extrusion in building["massing"]:
                polygons = [[(p[0], p[1]) for p in polygon] for polygon in extrusion["polygons"]]
                geometry = self.polygons_converter(polygons=polygons)
                geometry = shapely.make_valid(geometry, method='structure', keep_collapsed=False)
                footprints.append(geometry)
        return shapely.unary_union(footprints) if len(footprints) > 0 else shapely.GeometryCollection()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        footprint = self._footprint(massing=sample[self.massing_key])
        hull_area = shapely.area(footprint.convex_hull)
        if hull_area <= 0:
            return 0.0

        ratio = 1.0 - shapely.area(footprint) / hull_area
        return float(np.clip(ratio, 0.0, 1.0))
