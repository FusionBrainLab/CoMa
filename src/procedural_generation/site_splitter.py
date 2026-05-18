"""Split a site polygon into non-overlapping sub-regions for multiple buildings."""
import numpy as np
from shapely.geometry import Polygon, box, MultiPolygon
from shapely import affinity


class SiteSplitter:
    """Splits a site contour into non-overlapping sub-regions weighted by required footprint areas.

    Args:
        rectangular: If True, use treemap-based rectangular partitioning aligned
            to the site's oriented bounding rectangle, and fit rectangular
            footprints inside each sub-region.  If False (default), use the
            original polygon-bisection approach.
    """

    def __init__(self, rectangular=False):
        self.rectangular = rectangular

    def split_site(self, site_contour, requirements):
        """
        Split a site polygon into sub-regions, one per requirement.

        Each sub-region is sized proportionally to the building's required
        footprint area (usable_area / n_floors). Buildings are placed inside
        their sub-region without overlapping neighbours.

        Args:
            site_contour: List of polygons (list of list of [x, y]).
                          Typically a single outer polygon.
            requirements: List of requirement dicts with usable_area and n_floors.

        Returns:
            List of footprint polygons (list of [x, y] coords), one per requirement,
            each fitted inside a non-overlapping partition of the site.
        """
        if not requirements:
            return []

        site_poly = Polygon(site_contour[0])
        if not site_poly.is_valid:
            site_poly = site_poly.buffer(0)

        n = len(requirements)

        # Compute required footprint areas
        footprint_areas = []
        for req in requirements:
            # usable = req.get("usable_area", 100.0)
            usable = req.get("total_area", 100.0)
            floors = req.get("n_floors", 1)
            if floors <= 0:
                floors = 1
            if usable <= 0:
                usable = 100.0
            footprint_areas.append(usable / floors)

        if n == 1:
            # Single building - use region-scaling approach for exact area
            return [self._fit_footprint_in_region(site_poly, requirements[0])]

        total_required = sum(footprint_areas)
        weights = [a / total_required for a in footprint_areas]

        if self.rectangular:
            # Orthogonal treemap partitioning + polygon-scaling footprint fitting
            # This combines rectangular/orthogonal regions with zero-overlap
            # guarantees and exact area matching
            regions = self._partition_into_rectangles(site_poly, weights)
            return [
                self._fit_footprint_in_region(region, req)
                for region, req in zip(regions, requirements)
            ]

        # Original polygon-bisection path
        regions = self._partition_polygon(site_poly, weights)
        return [
            self._fit_footprint_in_region(region, req)
            for region, req in zip(regions, requirements)
        ]

    # ------------------------------------------------------------------
    # Rectangular-mode methods
    # ------------------------------------------------------------------

    def _partition_into_rectangles(self, polygon, weights):
        """Partition using the oriented bounding rectangle for rectangular sub-regions.

        Aligns the partitioning grid to the minimum rotated rectangle of the
        site, producing rectangular sub-regions that follow the site's natural
        orientation.  Each rectangle is then clipped to the actual site polygon.
        """
        obr = polygon.minimum_rotated_rectangle
        obr_coords = list(obr.exterior.coords)

        # Rotation angle from the OBR's first edge
        edge = np.array(obr_coords[1]) - np.array(obr_coords[0])
        angle = np.degrees(np.arctan2(edge[1], edge[0]))

        centroid = polygon.centroid
        rotated_poly = affinity.rotate(polygon, -angle, origin=centroid)

        minx, miny, maxx, maxy = rotated_poly.bounds
        rects = self._treemap_partition(minx, miny, maxx, maxy, weights)

        regions = []
        for rect in rects:
            clipped = rotated_poly.intersection(rect)
            clipped = _ensure_polygon(clipped)
            if clipped is None or clipped.is_empty:
                clipped = rect
            # Rotate back to original coordinate system
            region = affinity.rotate(clipped, angle, origin=centroid)
            region = _ensure_polygon(region)
            if region is None or region.is_empty:
                region = polygon
            regions.append(region)
        return regions

    def _treemap_partition(self, minx, miny, maxx, maxy, weights):
        """Recursively partition an axis-aligned rectangle into sub-rectangles
        whose areas are proportional to *weights*."""
        if len(weights) == 1:
            return [box(minx, miny, maxx, maxy)]

        mid = len(weights) // 2
        left_w, right_w = weights[:mid], weights[mid:]
        ratio = sum(left_w) / (sum(left_w) + sum(right_w))

        width = maxx - minx
        height = maxy - miny

        if width >= height:
            split = minx + width * ratio
            left = self._treemap_partition(minx, miny, split, maxy, left_w)
            right = self._treemap_partition(split, miny, maxx, maxy, right_w)
        else:
            split = miny + height * ratio
            left = self._treemap_partition(minx, miny, maxx, split, left_w)
            right = self._treemap_partition(minx, split, maxx, maxy, right_w)

        return left + right

    # ------------------------------------------------------------------
    # Original polygon-bisection methods
    # ------------------------------------------------------------------

    def _fit_footprint_in_region(self, region, requirement):
        """
        Create a footprint polygon inside the given region matching the required area.

        The footprint is the region shape scaled down (from its centroid) to match
        usable_area / n_floors. If that target area exceeds the region, we use the
        full region.

        Returns:
            List of [x, y] coordinates.
        """
        # usable = requirement.get("usable_area", 100.0)
        usable = requirement.get("total_area", 100.0)
        floors = requirement.get("n_floors", 1)
        if floors <= 0:
            floors = 1
        if usable <= 0:
            usable = 100.0
        target_area = usable / floors

        region_area = region.area
        if region_area <= 0:
            # Degenerate region – return a small square at centroid
            cx, cy = region.centroid.coords[0]
            side = np.sqrt(target_area)
            return [[cx - side/2, cy - side/2],
                    [cx + side/2, cy - side/2],
                    [cx + side/2, cy + side/2],
                    [cx - side/2, cy + side/2]]

        if target_area >= region_area:
            return self._polygon_to_coords(region)

        scale = np.sqrt(target_area / region_area)
        scaled = affinity.scale(region, xfact=scale, yfact=scale, origin="centroid")
        return self._polygon_to_coords(scaled)

    def _partition_polygon(self, polygon, weights):
        """
        Recursively bisect *polygon* into len(weights) pieces whose areas
        are proportional to *weights*.

        Returns:
            List of shapely Polygons, one per weight entry.
        """
        if len(weights) == 1:
            return [polygon]

        # Split weights into two roughly-equal groups
        mid = len(weights) // 2
        left_weights = weights[:mid]
        right_weights = weights[mid:]
        left_total = sum(left_weights)
        right_total = sum(right_weights)
        left_ratio = left_total / (left_total + right_total)

        left_poly, right_poly = self._bisect_polygon(polygon, left_ratio)

        left_parts = self._partition_polygon(left_poly, left_weights)
        right_parts = self._partition_polygon(right_poly, right_weights)

        return left_parts + right_parts

    def _bisect_polygon(self, polygon, ratio):
        """
        Split a polygon into two parts along its longest bounding-box axis
        so that the left part has approximately *ratio* of the total area.

        Uses binary search on the cutting position.

        Returns:
            (left_polygon, right_polygon)
        """
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny
        total_area = polygon.area
        target_area = total_area * ratio

        # Choose split axis: 0 = x (vertical cut), 1 = y (horizontal cut)
        if width >= height:
            axis = 0
            lo, hi = minx, maxx
        else:
            axis = 1
            lo, hi = miny, maxy

        # Binary search for the split position
        for _ in range(40):
            mid_val = (lo + hi) / 2.0
            left, right = self._cut_polygon(polygon, axis, mid_val)
            left_area = left.area if left is not None else 0
            if abs(left_area - target_area) < total_area * 1e-6:
                break
            if left_area < target_area:
                lo = mid_val
            else:
                hi = mid_val

        left, right = self._cut_polygon(polygon, axis, (lo + hi) / 2.0)

        # Ensure neither side is empty
        if left is None or left.is_empty:
            left = polygon
            right = polygon.centroid.buffer(0.01)
        if right is None or right.is_empty:
            right = polygon
            left = polygon.centroid.buffer(0.01)

        return left, right

    @staticmethod
    def _cut_polygon(polygon, axis, value):
        """
        Cut polygon with a line perpendicular to *axis* at *value*.

        Returns:
            (left_piece, right_piece) as shapely geometries.
        """
        minx, miny, maxx, maxy = polygon.bounds
        eps = max(maxx - minx, maxy - miny) * 0.01  # small buffer

        if axis == 0:  # vertical cut
            left_box = box(minx - eps, miny - eps, value, maxy + eps)
            right_box = box(value, miny - eps, maxx + eps, maxy + eps)
        else:  # horizontal cut
            left_box = box(minx - eps, miny - eps, maxx + eps, value)
            right_box = box(minx - eps, value, maxx + eps, maxy + eps)

        left = polygon.intersection(left_box)
        right = polygon.intersection(right_box)

        # Ensure we return Polygon (pick largest if MultiPolygon)
        left = _ensure_polygon(left)
        right = _ensure_polygon(right)

        return left, right

    @staticmethod
    def _polygon_to_coords(poly):
        """Extract exterior coordinates from a shapely polygon as list of [x, y]."""
        poly = _ensure_polygon(poly)
        if poly is None or poly.is_empty:
            return [[0, 0], [1, 0], [1, 1], [0, 1]]
        return [[float(p[0]), float(p[1])] for p in poly.exterior.coords[:-1]]


def _square_coords(cx, cy, side):
    """Return a list of [x, y] corners for an axis-aligned square centred at (cx, cy)."""
    hs = side / 2
    return [[cx - hs, cy - hs],
            [cx + hs, cy - hs],
            [cx + hs, cy + hs],
            [cx - hs, cy + hs]]


def _ensure_polygon(geom):
    """Return the largest Polygon from a geometry, or the geometry itself if already a Polygon."""
    if geom is None or geom.is_empty:
        return geom
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    # GeometryCollection or other – try to extract polygons
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    if polys:
        return max(polys, key=lambda g: g.area)
    return None
