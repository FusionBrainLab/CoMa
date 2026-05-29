from typing import Any, Dict

import shapely

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter


class MassingCoverage(SampleMetric):
    def __init__(self, *, massing_key: str,
                        site_key: str) -> None:
        self.massing_key = massing_key
        self.site_key = site_key
        self.polygons_converter = PolygonsToShapelyConverter()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        site = [[(p[0], p[1]) for p in polygon] for polygon in sample[self.site_key]]
        site = self.polygons_converter(polygons=site)
        site_area = shapely.area(site)
        if site_area <= 0:
            return 0.0

        footprints = []
        for building in sample[self.massing_key]:
            for extrusion in building["massing"]:
                polygons = [[(p[0], p[1]) for p in polygon] for polygon in extrusion["polygons"]]
                footprints.append(self.polygons_converter(polygons=polygons))
        if len(footprints) == 0:
            return 0.0

        footprint_area = shapely.area(shapely.unary_union(footprints))
        return footprint_area / site_area
