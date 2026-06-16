import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shapely

from .sample_metric import SampleMetric
from ..dataset_creator import DatasetCreator
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

# Catalogue of all available per-building features.
#   key: feature name
#   index: column position in the full 9-feature vector
FEATURE_CATALOG = {
    "log_area":     0,   # log(footprint area + 1)
    "log_perimeter":1,   # log(footprint perimeter + 1)
    "height":       2,   # total building height (top - bottom elevation)
    "circularity":  3,   # 4πA / P²
    "convex_rate":  4,   # area / convex-hull area
    "elongation":   5,   # 1 - minor_eigenvalue / major_eigenvalue of footprint cloud
    "aspect_ratio": 6,   # sqrt(major / minor) eigenvalue ratio
    "sin2theta":    7,   # sin(2 * orientation_angle)  -- folds 180° symmetry
    "cos2theta":    8,   # cos(2 * orientation_angle)
}
NUM_FEATURES_FULL = len(FEATURE_CATALOG)

# Named subsets for convenience.
FEATURE_SUBSETS = {
    "all":          list(FEATURE_CATALOG.keys()),
    "size":         ["log_area", "log_perimeter", "height"],
    "shape":        ["circularity", "convex_rate", "elongation", "aspect_ratio"],
    "orientation":  ["sin2theta", "cos2theta"],
    "no_orientation": ["log_area", "log_perimeter", "height",
                       "circularity", "convex_rate", "elongation", "aspect_ratio"],
    "classic":      ["log_area", "height", "circularity", "convex_rate",
                     "elongation", "sin2theta", "cos2theta"],   # original v1
}


class MassingContextFrechetDistance(SampleMetric):
    """FID-like contextual-relevance metric.

    Compares the *distribution of building features* inside the target massing
    against the distribution of building features across all context massings.
    Implemented as a diagonal Fréchet distance (sum of per-feature squared
    Wasserstein-2 distances between 1-D Gaussians), returned as exp(-frechet).

    Per-building features (see FEATURE_CATALOG):
      log_area, log_perimeter, height, circularity, convex_rate, elongation,
      aspect_ratio, sin(2θ), cos(2θ)

    Parameters
    ----------
    feature_subset : str or list[str]
        Which features to use.  Either a named preset from FEATURE_SUBSETS
        (``"all"``, ``"size"``, ``"shape"``, ``"orientation"``,
        ``"no_orientation"``, ``"classic"``) or an explicit list of feature
        names from FEATURE_CATALOG.
    """

    def __init__(self, *, massing_key: str,
                        context_key: str,
                        id_key: str,
                        context_dataset_loader: DatasetCreator,
                        feature_subset: Any = "all",
                        eps: float = 1e-9) -> None:
        self.massing_key = massing_key
        self.context_key = context_key
        self.id_key = id_key
        self.context_dataset_loader = context_dataset_loader
        self.eps = eps
        self.converter = PolygonsToShapelyConverter()

        if isinstance(feature_subset, str):
            if feature_subset not in FEATURE_SUBSETS:
                raise ValueError(f"Unknown feature_subset '{feature_subset}'. "
                                 f"Available: {list(FEATURE_SUBSETS)}")
            self.feature_names = FEATURE_SUBSETS[feature_subset]
        else:
            self.feature_names = list(feature_subset)
        for fn in self.feature_names:
            if fn not in FEATURE_CATALOG:
                raise ValueError(f"Unknown feature '{fn}'. Available: {list(FEATURE_CATALOG)}")
        self.feature_indices = [FEATURE_CATALOG[fn] for fn in self.feature_names]

        context_dataset = pd.DataFrame(self.context_dataset_loader())
        ids = context_dataset[self.id_key].tolist()
        massings = context_dataset[self.massing_key].tolist()
        self.context_features: Dict[str, np.ndarray] = {}
        for i in range(len(ids)):
            self.context_features[ids[i]] = self._massing_building_features(massings[i])
        del context_dataset

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _building_feature_full(self, building) -> np.ndarray:
        footprints = []
        top = -math.inf
        bottom = math.inf
        for extrusion in building["massing"]:
            polygons = [[(p[0], p[1]) for p in polygon] for polygon in extrusion["polygons"]]
            footprints.append(self.converter(polygons=polygons))
            top = max(top, extrusion["top_elevation"])
            bottom = min(bottom, extrusion["bottom_elevation"])
        footprint = shapely.unary_union(footprints)

        area = float(shapely.area(footprint))
        perimeter = float(footprint.length)
        if area <= 0 or perimeter <= 0:
            return np.full(NUM_FEATURES_FULL, np.nan)

        circularity = (4 * math.pi * area) / (perimeter ** 2)
        convex_hull_area = float(shapely.area(shapely.convex_hull(footprint)))
        convex_rate = area / convex_hull_area if convex_hull_area > 0 else np.nan
        height = float(top - bottom) if math.isfinite(top) and math.isfinite(bottom) else np.nan

        points = self._exterior_points(footprint)
        if len(points) >= 3:
            coords = np.array(points, dtype=float)
            coords = coords - coords.mean(axis=0)
            eig = np.linalg.eigvalsh(np.cov(coords, rowvar=False))
            major, minor = float(np.max(eig)), float(np.min(eig))
            elongation = float(np.clip(1.0 - minor / major, 0.0, 1.0)) if major > 0 else 0.0
            aspect_ratio = float(np.sqrt(major / (minor + self.eps))) if minor >= 0 else np.nan
        else:
            elongation = 0.0
            aspect_ratio = np.nan

        theta = self._orientation(footprint)

        return np.array([
            math.log(area + 1.0),
            math.log(perimeter + 1.0),
            height if height is not None else np.nan,
            circularity,
            convex_rate,
            elongation,
            aspect_ratio,
            math.sin(2 * theta),
            math.cos(2 * theta),
        ], dtype=float)

    def _exterior_points(self, geometry) -> list:
        if isinstance(geometry, shapely.geometry.Polygon):
            return [(p[0], p[1]) for p in geometry.exterior.coords[:-1]]
        if isinstance(geometry, shapely.geometry.MultiPolygon):
            return [pt for g in geometry.geoms for pt in self._exterior_points(g)]
        return []

    def _orientation(self, footprint) -> float:
        rect = footprint.minimum_rotated_rectangle
        if not hasattr(rect, "exterior"):
            return 0.0
        coords = list(rect.exterior.coords)[:-1]
        best_len = -1.0
        angle = 0.0
        for i in range(len(coords)):
            p1, p2 = coords[i], coords[(i + 1) % len(coords)]
            length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if length > best_len:
                best_len = length
                angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        return angle

    def _massing_building_features(self, massing) -> np.ndarray:
        vectors = []
        for building in massing:
            try:
                v = self._building_feature_full(building)
                vectors.append(v[self.feature_indices])
            except Exception:
                continue
        if not vectors:
            return np.empty((0, len(self.feature_indices)))
        return np.vstack(vectors)

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        target = self._massing_building_features(sample[self.massing_key])

        context_blocks = [
            self.context_features[i]
            for i in sample[self.context_key]
            if i in self.context_features
        ]
        context_blocks = [b for b in context_blocks if b.shape[0] > 0]
        if not context_blocks or target.shape[0] == 0:
            raise ValueError("empty target or context buildings")
        context = np.vstack(context_blocks)

        # keep columns with enough finite values
        col_ok = (
            (np.sum(np.isfinite(target), axis=0) >= 1) &
            (np.sum(np.isfinite(context), axis=0) >= 2)
        )
        if not np.any(col_ok):
            raise ValueError("no usable features")
        target = target[:, col_ok]
        context = context[:, col_ok]

        mu_c = np.nanmean(context, axis=0)
        sd_c = np.nanstd(context, axis=0) + self.eps
        zt = (target - mu_c) / sd_c
        zc = (context - mu_c) / sd_c

        mu_t = np.nanmean(zt, axis=0)
        sd_t = np.nanstd(zt, axis=0)
        mu_ctx = np.nanmean(zc, axis=0)
        sd_ctx = np.nanstd(zc, axis=0)

        frechet = float(np.nansum((mu_t - mu_ctx) ** 2 + (sd_t - sd_ctx) ** 2))
        if not math.isfinite(frechet):
            raise ValueError("non-finite frechet")
        return math.exp(-frechet)
