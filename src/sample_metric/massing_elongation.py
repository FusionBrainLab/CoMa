from typing import Any, Dict

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter


class MassingElongation(SampleMetric):
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

    def _exterior_points(self, *, geometry: shapely.Geometry):
        if isinstance(geometry, Polygon):
            return [(p[0], p[1]) for p in geometry.exterior.coords[:-1]]
        if isinstance(geometry, MultiPolygon):
            return [point for polygon in geometry.geoms for point in self._exterior_points(geometry=polygon)]
        return []

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        points = self._exterior_points(geometry=self._footprint(massing=sample[self.massing_key]))
        if len(points) < 3:
            return 0.0

        coords = np.array(points, dtype=float)
        coords = coords - np.mean(coords, axis=0)
        covariance = np.cov(coords, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        major = float(np.max(eigenvalues))
        minor = float(np.min(eigenvalues))
        if major <= 0:
            return 0.0

        elongation = 1.0 - minor / major
        return float(np.clip(elongation, 0.0, 1.0))
