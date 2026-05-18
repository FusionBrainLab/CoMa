"""Post-process generated 3D meshes to the massing format used in the dataset."""
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely import affinity


class MeshPostprocessor:
    """Extract building profiles from generated meshes and convert to massing format."""

    def extract_height_profile(self, mesh, n_samples=20):
        """
        Extract how the cross-sectional area of a mesh varies with height.

        Slices the mesh at *n_samples* equally-spaced Z-levels and measures
        the convex-hull area of the vertices near each slice.  The result is
        normalised so that the maximum value is 1.0.

        Args:
            mesh: trimesh.Trimesh object.
            n_samples: Number of vertical slices to take.

        Returns:
            List of *n_samples* floats in [0, 1] representing the relative
            cross-section scale at each height level.
        """
        vertices = mesh.vertices
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        z_range = z_max - z_min

        if z_range <= 0:
            return [1.0] * n_samples

        slice_height = z_range / n_samples
        areas = []

        for i in range(n_samples):
            z_lo = z_min + i * slice_height
            z_hi = z_lo + slice_height
            mask = (vertices[:, 2] >= z_lo) & (vertices[:, 2] <= z_hi)
            pts_2d = vertices[mask][:, :2]

            if len(pts_2d) >= 3:
                try:
                    hull = ConvexHull(pts_2d)
                    areas.append(hull.volume)  # 2-D ConvexHull: volume == area
                except Exception:
                    areas.append(0.0)
            else:
                areas.append(0.0)

        max_area = max(areas) if areas else 1.0
        if max_area <= 0:
            return [1.0] * n_samples

        profile = [a / max_area for a in areas]

        # Ensure the ground level is never smaller than the level above it
        # (buildings are typically wider at the base).
        for i in range(1, len(profile)):
            if profile[i] > profile[i - 1] and i <= len(profile) // 3:
                profile[i - 1] = profile[i]

        return profile

    def sample_profile_for_floors(self, profile, n_floors):
        """
        Resample a continuous height profile to one value per floor.

        Args:
            profile: List of floats from ``extract_height_profile``.
            n_floors: Number of building floors.

        Returns:
            List of *n_floors* scale factors in (0, 1].
        """
        n_profile = len(profile)
        if n_floors == 1:
            return [max(profile)]

        floor_scales = []
        for floor_idx in range(n_floors):
            # Map floor midpoint to a position in the profile
            t = (floor_idx + 0.5) / n_floors
            p_idx = t * (n_profile - 1)
            lo = int(p_idx)
            hi = min(lo + 1, n_profile - 1)
            frac = p_idx - lo
            value = profile[lo] * (1 - frac) + profile[hi] * frac
            floor_scales.append(max(value, 0.05))  # clamp to avoid zero

        return floor_scales

    def scale_polygon(self, polygon, scale_factor):
        """
        Scale a polygon by *scale_factor* around its centroid.

        Args:
            polygon: List of [x, y] coordinates.
            scale_factor: Linear scale factor (area scales by factor squared).

        Returns:
            Scaled polygon as list of [x, y].
        """
        if abs(scale_factor - 1.0) < 1e-6:
            return polygon

        poly = Polygon(polygon)
        if not poly.is_valid or poly.is_empty:
            return polygon

        scaled = affinity.scale(poly, xfact=scale_factor, yfact=scale_factor, origin="centroid")
        return [[float(x), float(y)] for x, y in scaled.exterior.coords[:-1]]

    def build_massing(self, footprint, n_floors, floor_height, floor_scales, building_id):
        """
        Construct a massing dict from a base footprint and per-floor scale factors.

        Args:
            footprint: Base footprint polygon (list of [x, y]).
            n_floors: Number of floors.
            floor_height: Height per floor in metres.
            floor_scales: List of per-floor scale factors (from height profile).
            building_id: ID string for the building.

        Returns:
            Dict with id and massing fields in the dataset format.
        """
        massing_levels = []
        for i in range(n_floors):
            scale = floor_scales[i] if i < len(floor_scales) else floor_scales[-1]
            floor_polygon = self.scale_polygon(footprint, scale)

            massing_levels.append({
                "polygons": [floor_polygon],
                "bottom_elevation": float(i * floor_height),
                "top_elevation": float((i + 1) * floor_height),
            })

        return {"id": str(building_id), "massing": massing_levels}

    def build_uniform_massing(self, footprint, n_floors, floor_height, building_id):
        """
        Fallback: single-extrusion massing with no profile variation.

        Args:
            footprint: Base footprint polygon (list of [x, y]).
            n_floors: Number of floors.
            floor_height: Height per floor in metres.
            building_id: Building ID string.

        Returns:
            Massing dict identical to what the procedural simple strategy produces.
        """
        polygon = [[float(x), float(y)] for x, y in footprint]
        return {
            "id": str(building_id),
            "massing": [{
                "polygons": [polygon],
                "bottom_elevation": 0.0,
                "top_elevation": float(n_floors * floor_height),
            }],
        }
