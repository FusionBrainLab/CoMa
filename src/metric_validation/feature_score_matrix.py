"""FeatureScoreMatrix — turn a dataset + named SampleMetrics into a score matrix.

Each feature is a ``SampleMetric`` (the same classes used by the single-metric
``Validator`` flow), supplied as a zero-arg factory (hydra ``_partial_: true``) so
the expensive context-pool initialisation only happens on a cache miss. The result
is an all-finite matrix ``X`` (rows × features) with aligned labels ``y`` and groups
(massing ids for leak-free GroupKFold), cached to ``cache_path`` as ``.npz``.
"""

import gc
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..core.base import Function
from ..dataset_creator import DatasetCreator


class FeatureScoreMatrix(Function):
    def __init__(self, *, submit_loader: DatasetCreator,
                        feature_metrics: Dict[str, Any],
                        label_key: str,
                        group_key: str,
                        cache_path: str,
                        force_rescore: bool = False,
                        verbose: bool = True) -> None:
        self.submit_loader = submit_loader
        self.feature_metrics = feature_metrics
        self.label_key = label_key
        self.group_key = group_key
        self.cache_path = cache_path
        self.force_rescore = force_rescore
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _load_cache(self) -> Dict[str, Any]:
        d = np.load(self.cache_path, allow_pickle=True)
        return {"X": d["X"], "y": d["y"], "groups": d["groups"],
                "names": [str(s) for s in d["names"]]}

    def _score_metric(self, factory: Any, records: List[Dict[str, Any]]) -> np.ndarray:
        sample_metric = factory()
        vals = np.full(len(records), np.nan)
        for i in range(len(records)):
            try:
                vals[i] = float(sample_metric(sample=records[i]))
            except Exception:
                pass
        del sample_metric
        gc.collect()
        return vals

    def __call__(self) -> Dict[str, Any]:
        names = list(self.feature_metrics.keys())

        # Cache hit: reuse the existing matrix iff the feature set matches.
        if os.path.exists(self.cache_path) and not self.force_rescore:
            cached = self._load_cache()
            if cached["names"] == names:
                self._log("FeatureScoreMatrix: cache hit %s  X=%s"
                          % (self.cache_path, cached["X"].shape))
                return cached
            self._log("FeatureScoreMatrix: cache feature mismatch, rescoring")

        self._log("FeatureScoreMatrix: loading submit dataset ...")
        df = pd.DataFrame(self.submit_loader())
        y = df[self.label_key].to_numpy()
        ids = df[self.group_key].to_numpy()
        records = df.to_dict("records")
        self._log("  n=%d  pos=%d  neg=%d" % (len(df), int(y.sum()), int((1 - y).sum())))

        raw = {}
        for name in names:
            t0 = time.time()
            raw[name] = self._score_metric(self.feature_metrics[name], records)
            fin = np.isfinite(raw[name])
            self._log("  scored %-18s valid=%d (%.1fs)"
                      % (name, int(fin.sum()), time.time() - t0))

        X_all = np.column_stack([raw[n] for n in names])
        mask = np.all(np.isfinite(X_all), axis=1)
        X, yv, gv = X_all[mask], y[mask], ids[mask]
        self._log("  joint-finite rows: %d / %d" % (int(mask.sum()), len(df)))

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez(self.cache_path, X=X, y=yv, groups=gv, names=np.array(names))
        self._log("  saved cache -> %s  X=%s" % (self.cache_path, X.shape))
        return {"X": X, "y": yv, "groups": gv, "names": names}
