"""LearnedEnsembleExperiment — GroupKFold comparison of learned metric combiners.

Evaluates each configured model (LogReg / HistGBDT / CatBoost …) over each named
feature set with leak-free GroupKFold-by-massing-id, caches out-of-fold predictions
for ROC plots, and computes per-model feature importance (model-agnostic permutation
+ CatBoost native) on a chosen feature set. Mirrors the original ``Validator``
contract: a ``Function`` whose ``__call__`` runs everything and writes JSON artefacts.
"""

import json
import os
import time
from typing import Any, Dict

import numpy as np
from sklearn.base import clone

from ..core.base import Function
from . import _cv_utils as cv
from .feature_sets import resolve_feature_set


class LearnedEnsembleExperiment(Function):
    def __init__(self, *, models: Dict[str, Any],
                        feature_sets: Dict[str, Any],
                        output_path: str,
                        oof_path: str,
                        importance_path: str,
                        importance_feature_set: str = "all",
                        n_splits: int = 5,
                        verbose: bool = True) -> None:
        self.models = models
        self.feature_sets = feature_sets
        self.output_path = output_path
        self.oof_path = oof_path
        self.importance_path = importance_path
        self.importance_feature_set = importance_feature_set
        self.n_splits = n_splits
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _resolve(self, value: Any, names: list) -> list:
        return resolve_feature_set(value, names)

    def __call__(self, *, data: Dict[str, Any]) -> Dict[str, Any]:
        X, y, groups, names = data["X"], data["y"], data["groups"], data["names"]
        idx = {n: i for i, n in enumerate(names)}
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        feature_sets = {k: self._resolve(v, names) for k, v in self.feature_sets.items()}
        results = {
            "n_samples": int(len(y)),
            "protocol": "GroupKFold(%d) by massing id" % self.n_splits,
            "feature_sets": feature_sets,
            "table": {},
        }
        oof_store = {}

        self._log("  %-18s %-9s  AUC ± std        F1 ± std" % ("feature_set", "model"))
        self._log("  " + "-" * 64)
        for fs_name, fs in feature_sets.items():
            cols = [idx[n] for n in fs]
            Xs = X[:, cols]
            for model_name, model in self.models.items():
                make_clf = (lambda m=model: clone(m.estimator))
                t0 = time.time()
                res, oof = cv.grouped_cv(Xs, y, groups, make_clf,
                                         standardize=model.standardize,
                                         return_oof=True, n_splits=self.n_splits)
                dt = time.time() - t0
                results["table"]["%s/%s" % (fs_name, model_name)] = {
                    **res, "n_features": len(fs), "seconds": round(dt, 1)}
                self._log("  %-18s %-9s  %.4f ± %.4f  %.4f ± %.4f  (%.1fs)"
                          % (fs_name, model_name, res["auc"], res["auc_std"],
                             res["f1"], res["f1_std"], dt))
                if fs_name == self.importance_feature_set:
                    oof_store[model_name] = oof

        np.savez(self.oof_path, y=y, **{m: o for m, o in oof_store.items()})
        with open(self.output_path, "w") as fh:
            json.dump(results, fh, indent=2)
        self._log("  saved -> %s" % self.output_path)

        self._compute_importance(X, y, groups, names, idx, feature_sets)

        best = max(results["table"].items(), key=lambda kv: kv[1]["auc"])
        self._log("  best: %s  AUC=%.4f  F1=%.4f"
                  % (best[0], best[1]["auc"], best[1]["f1"]))
        return results

    def _compute_importance(self, X, y, groups, names, idx, feature_sets) -> None:
        fs = feature_sets.get(self.importance_feature_set, list(names))
        cols = [idx[n] for n in fs]
        Xs = X[:, cols]
        self._log("  computing feature importance on '%s' ..." % self.importance_feature_set)
        importances = {"features": fs, "permutation": {}, "native": {}}
        for model_name, model in self.models.items():
            if model.standardize:
                continue  # permutation importance is for the tree combiners
            make_clf = (lambda m=model: clone(m.estimator))
            importances["permutation"][model_name] = cv.permutation_importance(
                Xs, y, groups, fs, make_clf, n_splits=self.n_splits)
            est = clone(model.estimator)
            est.fit(Xs, y)
            if hasattr(est, "get_feature_importance"):
                importances["native"][model_name] = {
                    n: float(v) for n, v in zip(fs, est.get_feature_importance())}
            elif hasattr(est, "feature_importances_"):
                importances["native"][model_name] = {
                    n: float(v) for n, v in zip(fs, est.feature_importances_)}
            self._log("    %-9s importance done" % model_name)
        with open(self.importance_path, "w") as fh:
            json.dump(importances, fh, indent=2)
        self._log("  saved -> %s" % self.importance_path)
