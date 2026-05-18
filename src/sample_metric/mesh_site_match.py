from typing import Any, Dict

import numpy as np
import shapely
from shapely.geometry import MultiPoint
import trimesh

from .sample_metric import SampleMetric
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter


class MeshSiteMatch(SampleMetric):
    """Site match metric for mesh predictions.

    Computes: area(site ∩ predicted_bottom) / area(predicted_bottom)
    """

    def __init__(self, *, output_col: str) -> None:
        self.output_col = output_col
        self.polygons_converter = PolygonsToShapelyConverter()

    def _repair_geometry(self, geometry: Any) -> Any:
        if geometry is None or getattr(geometry, "is_empty", False):
            return geometry
        if getattr(geometry, "is_valid", False):
            return geometry
        if hasattr(shapely, "make_valid"):
            repaired = shapely.make_valid(geometry)
        else:
            repaired = geometry.buffer(0)
        if repaired is None or getattr(repaired, "is_empty", False):
            return repaired
        if getattr(repaired, "is_valid", False):
            return repaired
        return repaired.buffer(0)

    def _polygonal_area(self, geometry: Any) -> float:
        if geometry is None:
            return 0.0
        geom_type = getattr(geometry, "geom_type", None)
        if geom_type in {"Polygon", "MultiPolygon"}:
            return float(shapely.area(geometry))
        if hasattr(geometry, "geoms"):
            total = 0.0
            for geom in geometry.geoms:
                if getattr(geom, "geom_type", None) in {"Polygon", "MultiPolygon"}:
                    total += float(shapely.area(geom))
            return total
        return float(shapely.area(geometry))

    def _mesh_footprint(self, meshes: list[trimesh.Trimesh]) -> Any:
        vertices = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
        if vertices.size == 0:
            return None

        lowest_z = float(np.min(vertices[:, 2]))
        base_vertices = vertices[np.isclose(vertices[:, 2], lowest_z, atol=1e-6)]
        if base_vertices.shape[0] < 3:
            base_vertices = vertices

        footprint = MultiPoint(base_vertices[:, :2]).convex_hull
        return self._repair_geometry(footprint)

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        # Convert site contour to shapely polygon and repair invalid rings if needed.
        site = [[(p[0], p[1]) for p in polygon] for polygon in sample["site_contour"]]
        site = self._repair_geometry(self.polygons_converter(polygons=site))

        # Load meshes from the output_col (each item is a dict with "massing" key)
        meshes = [trimesh.load(m["massing"]) for m in sample[self.output_col]]
        pred_bottom = self._mesh_footprint(meshes)

        if pred_bottom is None or getattr(pred_bottom, "is_empty", False):
            return 0.0

        try:
            intersection = site.intersection(pred_bottom)
        except Exception:
            site = self._repair_geometry(site)
            pred_bottom = self._repair_geometry(pred_bottom)
            try:
                intersection = site.intersection(pred_bottom)
            except Exception:
                return 0.0

        pred_area = self._polygonal_area(pred_bottom)
        if pred_area <= 1e-12:
            return 0.0

        site_match = self._polygonal_area(intersection) / pred_area
        if str(site_match) == "nan":
            site_match = 0.0
        return float(site_match)
