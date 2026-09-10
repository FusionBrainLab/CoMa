"""LearnedEnsembleTrainer — fit a learned metric on a dataset and serialise it.

The "tune on the bench" half of the reusable-metric workflow. It scores the feature
pool on the training dataset (reusing ``FeatureScoreMatrix`` + its cache), optionally
reports an honest GroupKFold estimate of generalisation, then fits the chosen model on
*all* rows and writes a portable artifact (estimator + feature order + standardisation)
that ``LearnedEnsembleMetric`` loads to validate any new dataset.
"""

import os
import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import clone

from ..core.base import Function
from ..sample_metric.learned_ensemble_metric import save_ensemble_artifact
from . import _cv_utils as cv
from .feature_sets import resolve_feature_set


class LearnedEnsembleTrainer(Function):
    def __init__(self, *, feature_matrix: Any,
                        model: Any,
                        model_output_path: str,
                        feature_set: Optional[Any] = None,
                        report_cv: bool = True,
                        n_splits: int = 5,
                        verbose: bool = True) -> None:
        self.feature_matrix = feature_matrix
        self.model = model
        self.model_output_path = model_output_path
        self.feature_set = feature_set
        self.report_cv = report_cv
        self.n_splits = n_splits
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def __call__(self) -> Dict[str, Any]:
        data = self.feature_matrix()
        X, y, groups, names = data["X"], data["y"], data["groups"], data["names"]

        feats = resolve_feature_set(self.feature_set, names)
        cols = [names.index(f) for f in feats]
        Xs = X[:, cols]
        self._log("LearnedEnsembleTrainer: %d samples × %d features %s"
                  % (len(y), len(feats), feats))

        meta: Dict[str, Any] = {
            "n_train_samples": int(len(y)),
            "feature_names": feats,
            "model": type(self.model.estimator).__name__,
            "standardize": bool(self.model.standardize),
        }

        if self.report_cv:
            t0 = time.time()
            res = cv.grouped_cv(Xs, y, groups,
                                lambda: clone(self.model.estimator),
                                standardize=self.model.standardize,
                                n_splits=self.n_splits)
            meta["cv"] = {"protocol": "GroupKFold(%d) by id" % self.n_splits, **res}
            self._log("  CV: AUC=%.4f±%.4f  F1=%.4f±%.4f  (%.1fs)"
                      % (res["auc"], res["auc_std"], res["f1"], res["f1_std"],
                         time.time() - t0))

        mean = std = None
        Xfit = Xs
        if self.model.standardize:
            mean, std = Xs.mean(0), Xs.std(0) + 1e-9
            Xfit = (Xs - mean) / std
        estimator = clone(self.model.estimator)
        estimator.fit(Xfit, y)
        self._log("  fitted %s on all rows" % meta["model"])

        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        save_ensemble_artifact(self.model_output_path, estimator=estimator,
                               feature_names=feats, standardize=self.model.standardize,
                               mean=mean, std=std, metadata=meta)
        self._log("  saved learned metric -> %s" % self.model_output_path)
        return meta
