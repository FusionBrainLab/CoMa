"""LearnedEnsembleMetric — a trained metric combiner usable as a drop-in SampleMetric.

This is the *reusable* learned metric. A model trained on one dataset (see
``LearnedEnsembleTrainer``) is serialised to a small artifact (estimator + feature
order + optional standardisation). At inference it behaves like any other
``SampleMetric``: given a sample it computes the configured feature vector and
returns the model's positive-class probability, so it slots straight into the
standard ``SampleMetricROCAUC`` / ``Validator`` flow on a **new** dataset.

Feature extractors are supplied via config (the shared feature pool, as
``_partial_`` factories) — NOT baked into the artifact — so retargeting to a new
dataset only means pointing the feature pool's context loader at the new context.
Only the features the trained model actually uses are instantiated.
"""

import functools
from typing import Any, Dict

import joblib
import numpy as np

from .sample_metric import SampleMetric


def save_ensemble_artifact(path: str, *, estimator: Any, feature_names: Any,
                           standardize: bool, mean: Any, std: Any,
                           metadata: Any = None) -> None:
    joblib.dump({
        "estimator": estimator,
        "feature_names": list(feature_names),
        "standardize": bool(standardize),
        "mean": None if mean is None else np.asarray(mean, dtype=float),
        "std": None if std is None else np.asarray(std, dtype=float),
        "metadata": metadata or {},
    }, path)


def load_ensemble_artifact(path: str) -> Dict[str, Any]:
    return joblib.load(path)


class LearnedEnsembleMetric(SampleMetric):
    def __init__(self, *, feature_metrics: Dict[str, Any],
                        model_path: str,
                        sample_massing_key: str = "massing") -> None:
        art = load_ensemble_artifact(model_path)
        self.estimator = art["estimator"]
        self.feature_names = list(art["feature_names"])
        self.standardize = bool(art["standardize"])
        self.mean = None if art["mean"] is None else np.asarray(art["mean"], dtype=float)
        self.std = None if art["std"] is None else np.asarray(art["std"], dtype=float)
        self.sample_massing_key = sample_massing_key
        # Instantiate only the features the trained model needs (factories or instances).
        self.features = {}
        for name in self.feature_names:
            if name not in feature_metrics:
                raise KeyError("feature pool is missing required feature '%s'" % name)
            entry = feature_metrics[name]
            self.features[name] = entry() if isinstance(entry, functools.partial) else entry

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        if self.sample_massing_key != "massing":
            sample = {**sample, "massing": sample[self.sample_massing_key]}
        x = np.empty(len(self.feature_names), dtype=float)
        for i, name in enumerate(self.feature_names):
            x[i] = float(self.features[name](sample=sample))
        if not np.all(np.isfinite(x)):
            raise ValueError("non-finite feature vector")
        if self.standardize and self.mean is not None:
            x = (x - self.mean) / self.std
        return float(self.estimator.predict_proba(x.reshape(1, -1))[0, 1])
