"""Regenerate the two per-sample TreeSHAP waterfalls (Figure 11) with the paper's
compact metric symbols.

Picks one confidently-accepted genuine pair and one confidently-rejected foreign pair
from the benchmark, computes exact CatBoost TreeSHAP values (in log-odds space), and
renders a waterfall for each, writing ``11_shap_waterfall_{genuine,foreign}.png`` to the
DVC ``figures/`` dir and the paper images dir.

Run:
    conda run -n massing python experiments/benchmarks/coma/metrics_validation/make_shap_waterfalls.py
"""

import os
import sys

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from catboost import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from metric_labels import LABELS  # noqa: E402

MODEL_PATH = os.path.join(HERE, "models", "learned_ensemble_catboost_all17.joblib")
SCORES_PATH = os.path.join(HERE, "results", "scores.npz")


def out_paths(tag):
    return [
        os.path.join(HERE, "figures", f"11_shap_waterfall_{tag}.png"),
        os.path.join(REPO, "papers", "coma", "latex_draft", "images",
                     "metric-selection", f"11_shap_waterfall_{tag}.png"),
    ]


def main() -> None:
    art = joblib.load(MODEL_PATH)
    estimator = art["estimator"]
    feat_names = list(art["feature_names"])

    d = np.load(SCORES_PATH, allow_pickle=True)
    names = [str(s) for s in d["names"]]
    col = [names.index(n) for n in feat_names]
    X = d["X"][:, col].astype(float)
    y = d["y"].astype(int)

    proba = estimator.predict_proba(X)[:, 1]
    shap_raw = np.asarray(estimator.get_feature_importance(Pool(X), type="ShapValues"))
    shap_values, base = shap_raw[:, :-1], shap_raw[:, -1]
    labels = [LABELS.get(n, n) for n in feat_names]

    # most confident correct decisions of each class
    genuine_i = int(np.where(y == 1, proba, -np.inf).argmax())
    foreign_i = int(np.where(y == 0, proba, np.inf).argmin())

    plt.rcParams.update({"font.size": 12, "savefig.dpi": 150, "figure.dpi": 150})
    for tag, i, title in (
        ("genuine", genuine_i, "Per-sample SHAP explanation \u2014 Genuine context"),
        ("foreign", foreign_i, "Per-sample SHAP explanation \u2014 Foreign context"),
    ):
        expl = shap.Explanation(
            values=shap_values[i], base_values=float(base[i]),
            data=X[i], feature_names=labels,
        )
        shap.plots.waterfall(expl, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8.0, 5.2)
        plt.title(f"{title}  ($\\hat p$={proba[i]:.2f})", fontsize=12)
        fig.tight_layout()
        for out in out_paths(tag):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fig.savefig(out, bbox_inches="tight")
            print("wrote %s" % out)
        plt.close(fig)


if __name__ == "__main__":
    main()
