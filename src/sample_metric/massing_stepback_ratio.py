from typing import Any, Dict

import numpy as np
import shapely

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter


class MassingStepbackRatio(SampleMetric):
    def __init__(self, *, massing_key: str) -> None:
        self.massing_key = massing_key
        self.polygons_converter = PolygonsToShapelyConverter()

    def _level_footprint(self, *, massing, elevation_key: str, target_fn) -> shapely.Geometry:
        footprints = []
        for building in massing:
            extrusions = building["massing"]
            if len(extrusions) == 0:
                continue
            target_elevation = target_fn([e[elevation_key] for e in extrusions])
            for extrusion in extrusions:
                if extrusion[elevation_key] != target_elevation:
                    continue
                polygons = [[(p[0], p[1]) for p in polygon] for polygon in extrusion["polygons"]]
                footprints.append(self.polygons_converter(polygons=polygons))
        return shapely.unary_union(footprints) if len(footprints) > 0 else shapely.GeometryCollection()

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        massing = sample[self.massing_key]
        ground = self._level_footprint(massing=massing, elevation_key="bottom_elevation", target_fn=min)
        ground_area = shapely.area(ground)
        if ground_area <= 0:
            return 0.0

        top = self._level_footprint(massing=massing, elevation_key="top_elevation", target_fn=max)
        top_area = shapely.area(top)
        return float(np.clip(top_area / ground_area, 0.0, 1.0))
