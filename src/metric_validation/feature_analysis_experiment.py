"""FeatureAnalysisExperiment — per-feature diagnostics on the score matrix.

Produces solo ROC-AUC / best-F1 per feature, the Spearman correlation matrix,
GroupKFold permutation importance, leave-one-out AUC, and forward greedy selection.
The exploratory rankings use a lightweight injected model + few folds for speed; the
final reported ensemble scores come from ``LearnedEnsembleExperiment``.
"""

import json
import os
from typing import Any, Dict

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from ..core.base import Function
from . import _cv_utils as cv


class FeatureAnalysisExperiment(Function):
    def __init__(self, *, analysis_model: Any,
                        output_path: str,
                        n_splits: int = 3,
                        verbose: bool = True) -> None:
        self.analysis_model = analysis_model
        self.output_path = output_path
        self.n_splits = n_splits
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def __call__(self, *, data: Dict[str, Any]) -> Dict[str, Any]:
        X, y, groups, names = data["X"], data["y"], data["groups"], data["names"]
        make_clf = (lambda: clone(self.analysis_model.estimator))
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        results = {"n_samples": int(len(y)), "n_features": len(names), "features": names,
                   "analysis_protocol": "GroupKFold(%d) + lightweight model" % self.n_splits}

        # 1. solo AUC / F1
        solo = {}
        for i, name in enumerate(names):
            col = X[:, i]
            solo[name] = {"roc_auc": float(roc_auc_score(y, col)),
                          "f1": cv.optimal_f1(y, col)[0]}
        results["solo"] = solo
        self._log("  solo AUC (top):")
        for name, d in sorted(solo.items(), key=lambda kv: -kv[1]["roc_auc"])[:5]:
            self._log("    %-18s AUC=%.4f F1=%.4f" % (name, d["roc_auc"], d["f1"]))

        # 2. Spearman correlation
        corr, _ = spearmanr(X)
        corr = np.atleast_2d(corr)
        results["correlation"] = {
            names[i]: {names[j]: float(corr[i, j]) for j in range(len(names))}
            for i in range(len(names))}

        # 3. permutation importance
        self._log("  permutation importance ...")
        results["permutation_importance"] = cv.permutation_importance(
            X, y, groups, names, make_clf, n_splits=self.n_splits)

        # 4. leave-one-out
        full_auc, _ = cv.cv_auc(X, y, groups, make_clf, n_splits=self.n_splits)
        loo = {}
        for fi, name in enumerate(names):
            keep = [j for j in range(len(names)) if j != fi]
            m, _ = cv.cv_auc(X[:, keep], y, groups, make_clf, n_splits=self.n_splits)
            loo[name] = {"auc_without": m, "delta_vs_full": m - full_auc}
        results["leave_one_out"] = {"full_auc": full_auc, "per_feature": loo}

        # 5. forward selection
        self._log("  forward greedy selection ...")
        results["forward_selection"] = cv.forward_selection(
            X, y, groups, names, make_clf, stop_delta=0.0, n_splits=self.n_splits)

        with open(self.output_path, "w") as fh:
            json.dump(results, fh, indent=2)
        self._log("  saved -> %s" % self.output_path)
        return results
