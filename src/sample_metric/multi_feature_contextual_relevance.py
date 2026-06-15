from typing import List, Any, Dict, Literal

import numpy as np
import pandas as pd

from .sample_metric import SampleMetric
from ..dataset_creator import DatasetCreator


class MultiFeatureContextualRelevance(SampleMetric):
    """Multivariate generalisation of ``MassingContextualRelevance``.

    Instead of collapsing a single massing characteristic, it builds a *feature
    vector* from a list of characteristics and measures how well the target
    massing's vector fits the distribution of the context massings' vectors.

    Features are z-normalised per sample using the context's own per-feature mean
    and std, so heterogeneous features (e.g. orientation in degrees and
    circularity in [0, 1]) are comparable. The fit is scored by one of:

    - ``mean``        : Euclidean distance to the context centroid (the
                        multivariate analogue of the original ``mean`` reduction).
    - ``nearest``     : distance to the closest context vector (multivariate
                        analogue of the original ``nearest`` reduction; robust to
                        multi-modal contexts such as orientation grids).
    - ``knn``         : mean distance to the ``k`` closest context vectors.
    - ``mahalanobis`` : Mahalanobis distance to the context Gaussian — the
                        per-sample, FID-like membership score (accounts for
                        correlations between features).

    For a single feature, ``mean``/``mahalanobis`` reproduce the original ``mean``
    metric and ``nearest`` reproduces the original ``nearest`` metric (ROC-AUC is
    invariant to the monotone ``exp(-d)`` wrapping).
    """

    def __init__(self, *, massing_key: str,
                        context_key: str,
                        additional_features: List[str],
                        massing_metrics: List[SampleMetric],
                        id_key: str,
                        context_dataset_loader: DatasetCreator,
                        reduction: Literal["mean", "nearest", "knn", "mahalanobis"],
                        k: int = 5,
                        eps: float = 1e-9) -> None:
        self.massing_key = massing_key
        self.context_key = context_key
        self.additional_features = additional_features
        self.massing_metrics = massing_metrics
        self.id_key = id_key
        self.context_dataset_loader = context_dataset_loader
        self.reduction = reduction
        self.k = k
        self.eps = eps

        context_dataset = pd.DataFrame(self.context_dataset_loader())
        ids = context_dataset[self.id_key].tolist()
        massings = context_dataset[self.massing_key].tolist()
        extra = {f: context_dataset[f].tolist() for f in self.additional_features}

        num_features = len(self.massing_metrics)
        matrix = np.full((len(ids), num_features), np.nan, dtype=float)
        for i in range(len(ids)):
            context_sample = {
                self.massing_key: massings[i],
                **{f: extra[f][i] for f in self.additional_features}
            }
            matrix[i] = self._features(context_sample)
        self.context_id_to_row = {ids[i]: i for i in range(len(ids))}
        self.context_matrix = matrix
        del context_dataset

    def _features(self, sample: Dict[str, Any]) -> np.ndarray:
        vector = np.full(len(self.massing_metrics), np.nan, dtype=float)
        for j, metric in enumerate(self.massing_metrics):
            try:
                value = float(metric(sample=sample))
                if np.isfinite(value):
                    vector[j] = value
            except Exception:
                pass
        return vector

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        target_sample = {
            self.massing_key: sample[self.massing_key],
            **{f: sample[f] for f in self.additional_features}
        }
        target = self._features(target_sample)

        rows = [self.context_id_to_row[i] for i in sample[self.context_key]
                if i in self.context_id_to_row]
        if len(rows) == 0:
            raise ValueError("no context massings found")
        context = self.context_matrix[rows]

        # keep features that are valid on the target and have enough context support
        col_ok = np.isfinite(target) & (np.sum(np.isfinite(context), axis=0) >= 2)
        if not np.any(col_ok):
            raise ValueError("no usable features")
        target = target[col_ok]
        context = context[:, col_ok]

        # keep context rows that are fully observed on the selected features
        row_ok = np.all(np.isfinite(context), axis=1)
        context = context[row_ok]
        if context.shape[0] < 2:
            raise ValueError("not enough context vectors")

        mu = context.mean(axis=0)
        sd = context.std(axis=0) + self.eps
        z_target = (target - mu) / sd
        z_context = (context - mu) / sd

        if self.reduction == "mean":
            distance = float(np.linalg.norm(z_target))
        elif self.reduction == "nearest":
            distance = float(np.min(np.linalg.norm(z_context - z_target, axis=1)))
        elif self.reduction == "knn":
            dists = np.sort(np.linalg.norm(z_context - z_target, axis=1))
            kk = int(min(self.k, dists.shape[0]))
            distance = float(np.mean(dists[:kk]))
        elif self.reduction == "mahalanobis":
            cov = np.cov(z_context, rowvar=False)
            cov = np.atleast_2d(cov) + self.eps * np.eye(z_target.shape[0])
            try:
                inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(cov)
            distance = float(np.sqrt(max(z_target @ inv @ z_target, 0.0)))
        else:
            raise ValueError(self.reduction)

        return float(np.exp(-distance))
