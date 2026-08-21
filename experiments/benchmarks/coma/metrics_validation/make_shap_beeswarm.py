"""Regenerate the TreeSHAP beeswarm for the learned CatBoost metric showing ALL 17
features (the original figure truncated to 12).

Loads the trained artifact (``models/learned_ensemble_catboost_all17.joblib``) and the
cached 17-feature score matrix (``results/scores.npz``), computes exact TreeSHAP values
via CatBoost's native ShapValues, and writes ``08_shap_beeswarm.png`` to both the
DVC-tracked ``figures/`` dir and the paper's image dir.

Run:
    conda run -n massing python experiments/benchmarks/coma/metrics_validation/make_shap_beeswarm.py
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
from metric_labels import LABELS  # noqa: E402  (compact symbols, single source of truth)

MODEL_PATH = os.path.join(HERE, "models", "learned_ensemble_catboost_all17.joblib")
SCORES_PATH = os.path.join(HERE, "results", "scores.npz")
OUT_PATHS = [
    os.path.join(HERE, "figures", "08_shap_beeswarm.png"),
    os.path.join(REPO, "papers", "coma", "images",
                 "metric-selection", "08_shap_beeswarm.png"),
    os.path.join(REPO, "papers", "coma", "latex_draft", "images",
                 "metric-selection", "08_shap_beeswarm.png"),
]


def main() -> None:
    art = joblib.load(MODEL_PATH)
    estimator = art["estimator"]
    feat_names = list(art["feature_names"])
    standardize = bool(art["standardize"])
    mean, std = art["mean"], art["std"]
    print("model: %s | %d features | standardize=%s"
          % (type(estimator).__name__, len(feat_names), standardize))

    d = np.load(SCORES_PATH, allow_pickle=True)
    X_all = d["X"]
    names = [str(s) for s in d["names"]]
    # align the score-matrix columns to the model's feature order
    col = [names.index(n) for n in feat_names]
    X = X_all[:, col].astype(float)
    print("score matrix: X=%s aligned to model feature order" % (X.shape,))

    # CatBoost was trained without standardisation; guard anyway
    X_model = X
    if standardize and mean is not None:
        X_model = (X - np.asarray(mean)) / np.asarray(std)

    # exact TreeSHAP via CatBoost native ShapValues -> (n, n_feat + 1); drop bias col
    shap_raw = estimator.get_feature_importance(Pool(X_model), type="ShapValues")
    shap_values = np.asarray(shap_raw)[:, :-1]
    print("shap values: %s" % (shap_values.shape,))

    labels = [LABELS.get(n, n) for n in feat_names]

    plt.rcParams.update({"font.size": 12, "savefig.dpi": 150, "figure.dpi": 150})
    # colour encodes the underlying feature value (each feature *is* a metric score)
    shap.summary_plot(
        shap_values, X, feature_names=labels,
        max_display=len(feat_names),          # show ALL 17
        color_bar_label="Metric score",
        sort=True, show=False, plot_size=(10.0, 7.5),
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel("SHAP value  ($\\leftarrow$ foreign context  |  genuine context $\\rightarrow$)")
    ax.set_title("Per-feature contributions to the learned metric (CatBoost, 17 features)",
                 fontsize=12, pad=12)
    fig.tight_layout()

    for out in OUT_PATHS:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print("wrote %s" % out)
    plt.close(fig)


if __name__ == "__main__":
    main()
