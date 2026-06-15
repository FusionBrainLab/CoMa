"""MetricFiguresRenderer — render presentation figures from cached result JSONs.

Reads ``feature_analysis.json`` / ``model_comparison.json`` / ``feature_importance.json``
(+ the OOF npz) from ``results_dir`` and writes PNGs to ``figures_dir``. Pure
plotting, no model fitting. Feature colour groups are config-supplied.
"""

import json
import os
from typing import Any, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score

from ..core.base import Function

GROUP_COLOR = {"scalar": "#4C72B0", "orientation_dist": "#DD8452", "extra_scalar": "#55A868"}
GROUP_LABEL = {"scalar": "base-9 scalar", "orientation_dist": "orientation distribution",
               "extra_scalar": "extra scalar"}
MODEL_COLOR = {"LogReg": "#8172B3", "GBDT": "#4C72B0", "CatBoost": "#C44E52"}


class MetricFiguresRenderer(Function):
    def __init__(self, *, results_dir: str,
                        figures_dir: str,
                        feature_groups: Dict[str, str],
                        verbose: bool = True) -> None:
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.feature_groups = feature_groups
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _read(self, name: str) -> Any:
        path = os.path.join(self.results_dir, name)
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return json.load(fh)

    def _color(self, name: str) -> str:
        return GROUP_COLOR.get(self.feature_groups.get(name, "scalar"), GROUP_COLOR["scalar"])

    def _save(self, fig: Any, name: str) -> None:
        plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
                             "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
        path = os.path.join(self.figures_dir, name)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self._log("  wrote %s" % path)

    def _group_legend(self, ax: Any) -> None:
        handles = [Line2D([0], [0], color=c, lw=8) for c in GROUP_COLOR.values()]
        ax.legend(handles, list(GROUP_LABEL.values()), loc="lower right", fontsize=9)

    def __call__(self) -> None:
        os.makedirs(self.figures_dir, exist_ok=True)
        fa = self._read("feature_analysis.json")
        mc = self._read("model_comparison.json")
        fi = self._read("feature_importance.json")
        self._log("MetricFiguresRenderer -> %s" % self.figures_dir)
        if fa:
            self._fig_solo_auc(fa)
            self._fig_correlation(fa)
            self._fig_forward_selection(fa)
            self._fig_permutation(fa)
        if mc:
            self._fig_model_comparison(mc)
            self._fig_roc(mc)
        if fi:
            self._fig_model_importance(fi)
        self._log("  done.")

    # ── 1. solo AUC ──
    def _fig_solo_auc(self, fa: Dict[str, Any]) -> None:
        solo = fa["solo"]
        items = sorted(solo.items(), key=lambda kv: kv[1]["roc_auc"])
        names = [k for k, _ in items]
        aucs = [v["roc_auc"] for _, v in items]
        baseline = solo.get("direction_nearest", {}).get("roc_auc", max(aucs))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(names, aucs, color=[self._color(n) for n in names])
        for i, a in enumerate(aucs):
            ax.text(a + 0.004, i, "%.3f" % a, va="center", fontsize=9)
        ax.axvline(0.5, color="grey", ls=":", lw=1)
        ax.axvline(baseline, color="#C44E52", ls="--", lw=1)
        ax.set_xlim(0.5, 0.92)
        ax.set_xlabel("ROC-AUC (solo metric)")
        ax.set_title("Single-metric contextual relevance — ROC-AUC")
        self._group_legend(ax)
        self._save(fig, "01_solo_auc.png")

    # ── 2. correlation heatmap ──
    def _fig_correlation(self, fa: Dict[str, Any]) -> None:
        names = fa["features"]
        corr = np.array([[fa["correlation"][a][b] for b in names] for a in names])
        fig, ax = plt.subplots(figsize=(9, 7.5))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=8)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, "%.2f" % corr[i, j], ha="center", va="center",
                        fontsize=6, color="black" if abs(corr[i, j]) < 0.6 else "white")
        ax.set_title("Feature correlation (Spearman)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.grid(False)
        self._save(fig, "02_correlation_heatmap.png")

    # ── 3. forward selection ──
    def _fig_forward_selection(self, fa: Dict[str, Any]) -> None:
        steps = fa["forward_selection"]
        x = [s["step"] for s in steps]
        auc = [s["auc"] for s in steps]
        err = [s["auc_std"] for s in steps]
        added = [s["added"] for s in steps]
        span = max(auc) - min(auc)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(x, auc, yerr=err, marker="o", color="#4C72B0", capsize=3, lw=2, zorder=3)
        for xi, ai, name in zip(x, auc, added):
            ax.annotate(name, (xi, ai + 0.0015), rotation=90, fontsize=8, ha="center", va="bottom")
        ax.set_xlabel("number of features (greedy forward selection)")
        ax.set_ylabel("GroupKFold ROC-AUC")
        ax.set_title("Forward feature selection — diminishing returns after ~6 features")
        ax.set_xticks(x)
        ax.set_ylim(min(auc) - 0.004, max(auc) + 0.6 * span)
        self._save(fig, "03_forward_selection.png")

    # ── 4. model comparison ──
    def _fig_model_comparison(self, mc: Dict[str, Any]) -> None:
        table = mc["table"]
        fsets = list(mc["feature_sets"].keys())
        models = [m for m in ["LogReg", "GBDT", "CatBoost"]
                  if any(k.endswith("/" + m) for k in table)]
        width = 0.8 / max(len(models), 1)
        xpos = np.arange(len(fsets))
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for mi, model in enumerate(models):
            aucs = [table["%s/%s" % (fs, model)]["auc"] for fs in fsets]
            errs = [table["%s/%s" % (fs, model)]["auc_std"] for fs in fsets]
            off = (mi - (len(models) - 1) / 2) * width
            bars = ax.bar(xpos + off, aucs, width, yerr=errs, capsize=3,
                          label=model, color=MODEL_COLOR.get(model))
            for b, a in zip(bars, aucs):
                ax.text(b.get_x() + b.get_width() / 2, a + 0.004, "%.3f" % a,
                        ha="center", fontsize=8)
        ax.set_xticks(xpos); ax.set_xticklabels(fsets, fontsize=9)
        ax.set_ylim(0.80, 0.93)
        ax.set_ylabel("GroupKFold ROC-AUC")
        ax.set_title("Learned ensembles: LogReg vs HistGBDT vs CatBoost")
        ax.legend(loc="upper left")
        self._save(fig, "04_model_comparison.png")

    # ── 5. ROC ──
    def _fig_roc(self, mc: Dict[str, Any]) -> None:
        path = os.path.join(self.results_dir, "oof_all17.npz")
        if not os.path.exists(path):
            return
        npz = np.load(path)
        y = npz["y"]
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        for model in ["LogReg", "GBDT", "CatBoost"]:
            if model not in npz.files:
                continue
            p = npz[model]
            m = np.isfinite(p)
            fpr, tpr, _ = roc_curve(y[m], p[m])
            ax.plot(fpr, tpr, lw=2, color=MODEL_COLOR.get(model),
                    label="%s (AUC=%.3f)" % (model, roc_auc_score(y[m], p[m])))
        ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("ROC — full-feature ensemble (out-of-fold)")
        ax.legend(loc="lower right")
        ax.set_aspect("equal")
        self._save(fig, "05_roc_all17.png")

    # ── 6. permutation importance ──
    def _fig_permutation(self, fa: Dict[str, Any]) -> None:
        pi = fa["permutation_importance"]
        items = sorted(pi.items(), key=lambda kv: kv[1]["mean_drop"])
        names = [k for k, _ in items]
        drops = [v["mean_drop"] for _, v in items]
        errs = [v["std_drop"] for _, v in items]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(names, drops, xerr=errs, color=[self._color(n) for n in names], capsize=2)
        ax.set_xlabel("AUC drop when feature is shuffled")
        ax.set_title("Permutation importance (analysis model)")
        self._group_legend(ax)
        self._save(fig, "06_permutation_importance.png")

    # ── 7. GBDT vs CatBoost importance ──
    def _fig_model_importance(self, fi: Dict[str, Any]) -> None:
        perm = fi.get("permutation", {})
        models = [m for m in ["GBDT", "CatBoost"] if m in perm]
        if len(models) < 1:
            return
        names = fi["features"]
        order = sorted(names, key=lambda n: sum(perm[m][n]["mean_drop"] for m in models))
        ypos = np.arange(len(order))
        h = 0.8 / len(models)
        fig, ax = plt.subplots(figsize=(9, 7))
        for mi, model in enumerate(models):
            vals = [perm[model][n]["mean_drop"] for n in order]
            errs = [perm[model][n]["std_drop"] for n in order]
            off = (mi - (len(models) - 1) / 2) * h
            ax.barh(ypos + off, vals, h, xerr=errs, capsize=2,
                    color=MODEL_COLOR.get(model), label=model)
        ax.set_yticks(ypos); ax.set_yticklabels(order, fontsize=9)
        ax.set_xlabel("permutation importance (ROC-AUC drop when feature is shuffled)")
        ax.set_title("Feature importance on the full ensemble — by model")
        ax.legend(loc="lower right")
        self._save(fig, "07_model_importance.png")
