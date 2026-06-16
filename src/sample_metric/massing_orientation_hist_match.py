"""Orientation-distribution contextual relevance.

The key observation is that a real massing tends to align with the *local
street grid*.  The context massings collectively encode that grid as a
multi-modal distribution of orientation angles.  A target massing that shares
one of those modes fits its context; one with a random orientation does not.

``Direction__nearest`` already exploits this, but it collapses each massing to
a *single* scalar angle (the orientation of its largest footprint) and then
picks the context building whose angle is nearest.  The resulting score depends
heavily on which building happens to dominate.

Here we take a **distribution-level** approach:

1.  Extract all per-building orientation angles from the *context* massings —
    angles are in ``[0, π)`` because of the 180° rotational symmetry of a
    building footprint.  Then map them to ``[0, π)`` using the ``2θ`` trick
    (double the angle, apply mod 2π), treating the half-circle as a full
    circle for KDE purposes.

2.  Fit a *von-Mises KDE* to those angles — the circular analogue of Gaussian
    KDE — with bandwidth selected by Silverman's rule on the circular
    distribution.

3.  Evaluate the log-density at each target building's angle, average over
    target buildings, and return ``exp(mean_log_density)``.

High score → target orientations lie at a peak of the context distribution →
contextually relevant.

Parameters
----------
bandwidth_factor : float
    Multiplier for the Silverman bandwidth. Larger → smoother KDE.
    Default 1.0 (standard Silverman).
"""

import math
from typing import Any, Dict

import numpy as np
import pandas as pd
import shapely

from .sample_metric import SampleMetric
from ..dataset_creator import DatasetCreator
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter


class MassingOrientationHistMatch(SampleMetric):
    """Contextual relevance via von-Mises KDE of orientation distribution."""

    def __init__(self, *, massing_key: str,
                        context_key: str,
                        id_key: str,
                        context_dataset_loader: DatasetCreator,
                        bandwidth_factor: float = 1.0,
                        eps: float = 1e-9) -> None:
        self.massing_key = massing_key
        self.context_key = context_key
        self.id_key = id_key
        self.bandwidth_factor = bandwidth_factor
        self.eps = eps
        self.converter = PolygonsToShapelyConverter()

        context_dataset = pd.DataFrame(context_dataset_loader())
        ids = context_dataset[id_key].tolist()
        massings = context_dataset[massing_key].tolist()
        # cache per-context-massing: array of orientation angles (one per building)
        self.context_angles: Dict[str, np.ndarray] = {}
        for i in range(len(ids)):
            self.context_angles[ids[i]] = self._massing_orientations(massings[i])
        del context_dataset

    # ------------------------------------------------------------------

    def _footprint_orientation(self, building) -> float:
        """Return dominant orientation angle in [0, π), or nan on failure."""
        try:
            footprints = []
            for extrusion in building["massing"]:
                polygons = [[(p[0], p[1]) for p in polygon]
                            for polygon in extrusion["polygons"]]
                footprints.append(self.converter(polygons=polygons))
            footprint = shapely.unary_union(footprints)
            if footprint.is_empty:
                return np.nan
            rect = footprint.minimum_rotated_rectangle
            if not hasattr(rect, "exterior"):
                return np.nan
            coords = list(rect.exterior.coords)[:-1]
            best_len, angle = -1.0, 0.0
            for j in range(len(coords)):
                p1, p2 = coords[j], coords[(j + 1) % len(coords)]
                length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                if length > best_len:
                    best_len = length
                    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            # fold into [0, π) to exploit 180° symmetry
            angle = angle % math.pi
            return float(angle)
        except Exception:
            return np.nan

    def _massing_orientations(self, massing) -> np.ndarray:
        angles = []
        for building in massing:
            a = self._footprint_orientation(building)
            if np.isfinite(a):
                angles.append(a)
        return np.array(angles, dtype=float)

    # ------------------------------------------------------------------
    # von-Mises KDE helpers (working on the *doubled* angle in [0, 2π))
    # ------------------------------------------------------------------

    @staticmethod
    def _silverman_kappa(angles_2pi: np.ndarray) -> float:
        """Approximate concentration parameter κ via Silverman's rule.

        We use the mean resultant length R̄ as a summary of spread:
          κ ≈ 2/σ²  where σ² ≈ 1 − R̄  (circular variance).
        Then apply Silverman's n^{-2/5} bandwidth scaling.
        """
        n = len(angles_2pi)
        if n < 2:
            return 1.0
        sin_mean = np.mean(np.sin(angles_2pi))
        cos_mean = np.mean(np.cos(angles_2pi))
        R_bar = math.sqrt(sin_mean ** 2 + cos_mean ** 2)
        R_bar = min(R_bar, 1.0 - 1e-9)
        # circular variance → bandwidth → concentration
        circ_var = 1.0 - R_bar
        sigma_sq = -2.0 * math.log(R_bar + 1e-9)  # von Mises approx
        # Silverman factor for circular data (analogous to normal rule-of-thumb)
        bw = (4 * math.pi ** 0.5 * math.exp(-sigma_sq / 2) /
              (3 * n)) ** (1 / 5)
        kappa = max(1.0 / (bw ** 2 + 1e-9), 0.5)
        return float(kappa)

    @staticmethod
    def _von_mises_log_density(target_angles: np.ndarray,
                               context_angles: np.ndarray,
                               kappa: float) -> np.ndarray:
        """Log-density of von-Mises KDE at each target angle.

        log p(θ) = log(1/n) + log∑ exp(κ cos(θ − θᵢ)) − log(2π I₀(κ))
        """
        # log(2π I₀(κ)) — numerically stable
        log_norm = math.log(2 * math.pi) + _log_i0(kappa)
        # Vectorised: diff shape (len_target, len_context)
        diff = target_angles[:, None] - context_angles[None, :]   # broadcast
        log_sum = _logsumexp(kappa * np.cos(diff), axis=1)        # (len_target,)
        return log_sum - math.log(len(context_angles)) - log_norm  # (len_target,)

    # ------------------------------------------------------------------

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        target_angles_raw = self._massing_orientations(sample[self.massing_key])
        if len(target_angles_raw) == 0:
            raise ValueError("target massing has no valid orientations")

        context_angle_lists = [
            self.context_angles[i]
            for i in sample[self.context_key]
            if i in self.context_angles
        ]
        context_angle_lists = [a for a in context_angle_lists if len(a) > 0]
        if not context_angle_lists:
            raise ValueError("no valid context orientations")

        # Collect all context building angles
        context_angles_raw = np.concatenate(context_angle_lists)

        # Double all angles to map [0, π) → [0, 2π)
        target_2pi = (2.0 * target_angles_raw) % (2 * math.pi)
        context_2pi = (2.0 * context_angles_raw) % (2 * math.pi)

        kappa = self._silverman_kappa(context_2pi) * self.bandwidth_factor
        log_dens = self._von_mises_log_density(target_2pi, context_2pi, kappa)
        # Average log-density over target buildings, exponentiate
        mean_log_dens = float(np.mean(log_dens))
        # Shift so that the expected score at the mode is ~1; clip to [0,1]
        return float(np.exp(mean_log_dens))


# ------------------------------------------------------------------
# Numerical helpers (avoid scipy dependency)
# ------------------------------------------------------------------

def _log_i0(kappa: float) -> float:
    """Log of modified Bessel function I₀(κ) — numerically stable approximation."""
    if kappa < 3.75:
        t = (kappa / 3.75) ** 2
        poly = (1.0 + t * (3.5156229 + t * (3.0899424 + t * (
                1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813))))))
        return math.log(max(poly, 1e-300))
    else:
        # asymptotic: I₀(κ) ≈ exp(κ) / sqrt(2πκ)
        return kappa - 0.5 * math.log(2 * math.pi * kappa)


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log-sum-exp along an axis."""
    xmax = np.max(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - xmax), axis=axis)) + xmax.squeeze(axis=axis)
