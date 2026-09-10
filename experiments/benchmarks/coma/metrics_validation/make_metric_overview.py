"""Regenerate the benchmark overview (Figure 10) with the paper's compact metric
symbols and the new family-based colouring.

Two panels --- ROC-AUC and best-threshold F1 --- with a top block of learned ensembles
over the nested feature sets (Core / Core+Dist / All) and a lower block of the 17 single
scores, sorted by ROC-AUC and coloured by characteristic family (orientation, scale &
density, shape) and by distribution metric. Reads ``results/consolidated_results.json``
and writes ``10_metric_overview.png`` to the DVC ``figures/`` dir and the paper images dir.

Run:
    conda run -n massing python experiments/benchmarks/coma/metrics_validation/make_metric_overview.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from metric_labels import LABELS, GROUP, GROUP_COLOR, GROUP_LABEL  # noqa: E402

RESULTS_PATH = os.path.join(HERE, "results", "consolidated_results.json")
OUT_PATHS = [
    os.path.join(HERE, "figures", "10_metric_overview.png"),
    os.path.join(REPO, "papers", "coma", "images",
                 "metric-selection", "10_metric_overview.png"),
    os.path.join(REPO, "papers", "coma", "latex_draft", "images",
                 "metric-selection", "10_metric_overview.png"),
]

# learned-ensemble rows, top-to-bottom, mapped to model_comparison keys
ENSEMBLES = [
    ("CatBoost (All)",        "all17/CatBoost"),
    ("GBDT (All)",            "all17/GBDT"),
    ("GBDT (Core+Dist)",      "base9_plus_orient/GBDT"),
    ("GBDT (Core)",           "base9/GBDT"),
    ("Logistic reg. (All)",   "all17/LogReg"),
]

GAP = 1.6  # vertical gap (rows) between the ensemble and single-score blocks


def main() -> None:
    with open(RESULTS_PATH) as fh:
        res = json.load(fh)
    solo = res["solo"]
    mc = res["model_comparison"]

    singles_desc = sorted(solo.items(), key=lambda kv: kv[1]["roc_auc"], reverse=True)

    # assign y positions bottom-to-top: singles first, then a gap, then ensembles
    rows = []  # (y, label, auc, f1, color, is_ensemble)
    y = 0.0
    for cid, vals in reversed(singles_desc):  # worst at the bottom
        rows.append((y, LABELS.get(cid, cid), vals["roc_auc"], vals["f1"],
                     GROUP_COLOR[GROUP[cid]], False))
        y += 1.0
    single_y = [r[0] for r in rows]
    y += GAP
    for label, key in reversed(ENSEMBLES):  # worst ensemble at the bottom of its block
        rows.append((y, label, mc[key]["auc"], mc[key]["f1"],
                     GROUP_COLOR["ensemble"], True))
        y += 1.0
    ens_y = [r[0] for r in rows if r[5]]

    best_single_auc = max(v["roc_auc"] for v in solo.values())
    best_single_f1 = max(v["f1"] for v in solo.values())

    plt.rcParams.update({"font.size": 12, "savefig.dpi": 150, "figure.dpi": 150})
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(11.0, 9.5), sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    def draw(ax, value_idx, title, xlabel, best):
        ys = [r[0] for r in rows]
        vals = [r[value_idx] for r in rows]
        colors = [r[4] for r in rows]
        ax.barh(ys, vals, color=colors, height=0.72, zorder=3)
        for yy, vv in zip(ys, vals):
            ax.text(vv + 0.006, yy, f"{vv:.3f}", va="center", ha="left",
                    fontsize=10, zorder=4)
        ax.axvline(best, ls="--", lw=1.2, color="#C0392B", zorder=2)
        ax.text(best - 0.004, single_y[len(single_y) // 2],
                f"best single {best:.3f}", rotation=90, va="center", ha="right",
                fontsize=9, color="#C0392B")
        ax.set_xlim(0.5, 0.97)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=13)
        ax.grid(axis="x", ls=":", lw=0.6, color="0.8", zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    draw(axa, 2, "(a) Ranking power (ROC-AUC)", "ROC-AUC", best_single_auc)
    draw(axb, 3, r"(b) Decision quality ($F_1$)", r"best-threshold $F_1$", best_single_f1)

    axa.set_yticks([r[0] for r in rows])
    axa.set_yticklabels([r[1] for r in rows])
    axa.set_ylim(-0.8, rows[-1][0] + 0.8)

    # block brackets + rotated side labels on the left of panel (a)
    tx = axa.get_yaxis_transform()  # x in axes fraction, y in data coords
    for ys, text in ((single_y, "SINGLE\nCHARACTERISTICS"), (ens_y, "LEARNED\nENSEMBLES")):
        y0, y1 = min(ys) - 0.4, max(ys) + 0.4
        axa.plot([-0.56, -0.56], [y0, y1], transform=tx, color="0.3", lw=1.4,
                 clip_on=False)
        axa.text(-0.60, (y0 + y1) / 2, text, transform=tx, rotation=90,
                 va="center", ha="center", fontsize=10, fontweight="bold",
                 color="0.25")

    legend_order = ["ensemble", "orientation", "distribution", "scale_density", "shape"]
    handles = [Patch(facecolor=GROUP_COLOR[g], label=GROUP_LABEL[g]) for g in legend_order]
    handles.append(Line2D([0], [0], ls="--", color="#C0392B", label="best single score"))
    axb.legend(handles=handles, loc="lower right", fontsize=10, frameon=True,
               framealpha=0.95)

    fig.tight_layout()
    for out in OUT_PATHS:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print("wrote %s" % out)
    plt.close(fig)


if __name__ == "__main__":
    main()
