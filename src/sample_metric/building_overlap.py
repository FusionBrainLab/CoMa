from typing import Any, Dict
from itertools import combinations

import shapely

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter


class BuildingOverlap(SampleMetric):
    """Measures average pairwise overlap between building footprints in a site.

    For each pair of buildings, computes intersection_area / min(area_a, area_b).
    Returns the mean across all pairs. Returns 0 if fewer than 2 buildings.
    """

    def __init__(self, *, output_col: str) -> None:
        self.output_col = output_col
        self.polygons_converter = PolygonsToShapelyConverter()

    def _get_building_footprints(self, *, sample: Dict[str, Any]) -> list:
        """Extract individual building footprints as shapely geometries."""
        pred_massings = sample[self.output_col]
        footprints = []
        for m in pred_massings:
            massing = m["massing"]
            if not massing:
                continue
            bottom_elevation = min(e["bottom_elevation"] for e in massing)
            bottoms = [e["polygons"] for e in massing if e["bottom_elevation"] == bottom_elevation]
            bottom_polys = []
            for foot in bottoms:
                bottom_polys.extend([[(p[0], p[1]) for p in polygon] for polygon in foot])
            if bottom_polys:
                footprints.append(self.polygons_converter(polygons=bottom_polys))
        return footprints

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        footprints = self._get_building_footprints(sample=sample)

        if len(footprints) < 2:
            return 0.0

        overlaps = []
        for fp_a, fp_b in combinations(footprints, 2):
            area_a = shapely.area(fp_a)
            area_b = shapely.area(fp_b)
            min_area = min(area_a, area_b)
            if min_area <= 0:
                continue
            intersection_area = shapely.area(fp_a.intersection(fp_b))
            overlaps.append(intersection_area / min_area)

        if not overlaps:
            return 0.0

        return sum(overlaps) / len(overlaps)
