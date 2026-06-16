"""MetricsValidationExperiment — top-level orchestrator (the config ``method``).

Runs the full metric-validation pipeline in the original repo style: a single
``Function`` whose ``__call__()`` executes each injected stage and writes all
artefacts under the experiment folder. Stages share one feature score matrix so the
expensive scoring happens at most once.

Pipeline: FeatureScoreMatrix -> FeatureAnalysisExperiment -> LearnedEnsembleExperiment
-> MetricFiguresRenderer -> consolidated results JSON + summary.
"""

import json
import os
from typing import Any, Dict, Optional

from ..core.base import Function


class MetricsValidationExperiment(Function):
    def __init__(self, *, feature_matrix: Any,
                        consolidated_path: str,
                        results_dir: str,
                        feature_analysis: Optional[Any] = None,
                        ensemble_experiment: Optional[Any] = None,
                        figures_renderer: Optional[Any] = None,
                        single_metric_sweep_path: Optional[str] = None,
                        verbose: bool = True) -> None:
        self.feature_matrix = feature_matrix
        self.consolidated_path = consolidated_path
        self.results_dir = results_dir
        self.feature_analysis = feature_analysis
        self.ensemble_experiment = ensemble_experiment
        self.figures_renderer = figures_renderer
        self.single_metric_sweep_path = single_metric_sweep_path
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _read(self, path: Optional[str]) -> Dict[str, Any]:
        if path and os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
        return {}

    def __call__(self) -> None:
        sep = "=" * 72
        self._log(sep + "\n[1/4] FEATURE SCORE MATRIX\n" + sep)
        data = self.feature_matrix()

        fa_res, mc_res = {}, {}
        if self.feature_analysis is not None:
            self._log(sep + "\n[2/4] FEATURE ANALYSIS\n" + sep)
            fa_res = self.feature_analysis(data=data)
        if self.ensemble_experiment is not None:
            self._log(sep + "\n[3/4] LEARNED ENSEMBLE COMPARISON\n" + sep)
            mc_res = self.ensemble_experiment(data=data)
        if self.figures_renderer is not None:
            self._log(sep + "\n[4/4] FIGURES\n" + sep)
            self.figures_renderer()

        self._consolidate(fa_res, mc_res)

    def _consolidate(self, fa: Dict[str, Any], mc: Dict[str, Any]) -> None:
        sep = "=" * 72
        fa = fa or self._read(os.path.join(self.results_dir, "feature_analysis.json"))
        mc = mc or self._read(os.path.join(self.results_dir, "model_comparison.json"))
        sweep = self._read(self.single_metric_sweep_path).get("negative_sampling_testset", {})
        solo = fa.get("solo", {})
        table = mc.get("table", {})

        consolidated = {
            "dataset": {
                "name": "CoMa negative-sampling validation set",
                "n_samples_scored": fa.get("n_samples"),
                "cv": "GroupKFold(5) by massing id (no positive/negative leakage)",
            },
            "single_metric_sweep": dict(sorted(sweep.items(), key=lambda kv: -kv[1])),
            "solo": dict(sorted(solo.items(), key=lambda kv: -kv[1]["roc_auc"])),
            "feature_analysis": {
                "correlation": fa.get("correlation", {}),
                "permutation_importance": fa.get("permutation_importance", {}),
                "leave_one_out": fa.get("leave_one_out", {}),
                "forward_selection": fa.get("forward_selection", []),
            },
            "model_comparison": table,
        }
        if solo:
            best_single = max(solo.items(), key=lambda kv: kv[1]["roc_auc"])
            consolidated["headline_best_single"] = {"name": best_single[0], **best_single[1]}
        if table:
            best_ens = max(table.items(), key=lambda kv: kv[1]["auc"])
            consolidated["headline_best_ensemble"] = {"name": best_ens[0], **best_ens[1]}

        os.makedirs(os.path.dirname(self.consolidated_path), exist_ok=True)
        with open(self.consolidated_path, "w") as fh:
            json.dump(consolidated, fh, indent=2)

        self._log("\n" + sep + "\nCONSOLIDATED SUMMARY\n" + sep)
        if solo:
            self._log("Top single metrics:")
            for name, d in list(consolidated["solo"].items())[:6]:
                self._log("  %-18s AUC=%.4f F1=%.4f" % (name, d["roc_auc"], d["f1"]))
        if table:
            self._log("\nLearned ensembles (sorted):")
            for key, d in sorted(table.items(), key=lambda kv: -kv[1]["auc"]):
                self._log("  %-26s AUC=%.4f F1=%.4f" % (key, d["auc"], d["f1"]))
        self._log("\nSaved -> %s" % self.consolidated_path)
