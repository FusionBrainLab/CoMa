"""Regenerate the two per-sample TreeSHAP waterfalls (Figure 11) with the paper's
compact metric symbols.

Picks one confidently-accepted genuine pair and one confidently-rejected foreign pair
from the benchmark, computes exact CatBoost TreeSHAP values (in log-odds space), and
renders a waterfall for each, writing ``11_shap_waterfall_{genuine,foreign}.{pgf,pdf,png}``
to the DVC ``figures/`` dir and the paper images dir.

The figures are authored at the MDPI ``\\textwidth`` so LaTeX includes them at 1:1 and
never shrinks the type, and are exported as vector pgf alongside the raster fallbacks.

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
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrow
from matplotlib.transforms import ScaledTranslation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from metric_labels import LABELS  # noqa: E402

MODEL_PATH = os.path.join(HERE, "models", "learned_ensemble_catboost_all17.joblib")
SCORES_PATH = os.path.join(HERE, "results", "scores.npz")

# MDPI single-column body width; authoring at this width keeps the rendered point sizes
# below identical to the point sizes the reader sees on paper.
TEXT_WIDTH_INCHES = 5.4567
FIGURE_HEIGHT_INCHES = 3.65
FORMATS = ("pgf", "pdf", "png")
EXPORT_DPI = 600.0

# shap hardcodes 12pt/13pt against an 8in canvas; these are those sizes brought down to
# the narrower canvas, which is what the reader used to get after LaTeX shrank the figure.
FEATURE_LABEL_SIZE = 8.0
BAR_VALUE_SIZE = 7.5
AXIS_TICK_SIZE = 7.5
ENDPOINT_LABEL_SIZE = 7.5
TITLE_SIZE = 8.5

# shap's secondary text is #999999, too light to survive print. These keep the same
# hierarchy (secondary text still recedes) at a legible contrast ratio.
SECONDARY_TEXT_COLOR = "#4d4d4d"
SECONDARY_TEXT_LUMINANCE = 0.5
OUTSIDE_LABEL_DARKENING = 0.76

PGF_PREAMBLE = [
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{lmodern}",
]


def out_paths(tag, extension):
    return [
        os.path.join(HERE, "figures", f"11_shap_waterfall_{tag}.{extension}"),
        os.path.join(REPO, "papers", "coma", "latex_draft", "images",
                     "metric-selection", f"11_shap_waterfall_{tag}.{extension}"),
    ]


def darken(color, factor):
    return tuple(channel * factor for channel in to_rgb(color))


def luminance(color):
    red, green, blue = to_rgb(color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def ascii_minus(text):
    """shap formats numbers with U+2212, which plain inputenc cannot typeset.

    Keeping the pgf free of it lets the paper ``\\input`` the figure without adding a
    ``\\DeclareUnicodeCharacter`` line to its preamble.
    """
    text.set_text(text.get_text().replace("\u2212", "-"))


def restyle_waterfall(fig, title):
    """Re-typeset a shap waterfall for print: legible sizes, print-safe contrast."""
    ax, base_axis, prediction_axis = fig.axes[0], fig.axes[1], fig.axes[2]
    fig.set_size_inches(TEXT_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)

    for text in ax.texts:
        text.set_fontsize(BAR_VALUE_SIZE)
        ascii_minus(text)
        # White labels sit on a saturated bar and already contrast; the rest are drawn
        # on the white background in the bar's own colour and need deepening.
        if to_rgb(text.get_color()) != (1.0, 1.0, 1.0):
            text.set_color(darken(text.get_color(), OUTSIDE_LABEL_DARKENING))

    for label in ax.get_yticklabels():
        label.set_fontsize(FEATURE_LABEL_SIZE)
        ascii_minus(label)
        # Every row carries two overlaid labels: the feature name in black over the
        # feature value in grey. Only the grey one needs deepening.
        if luminance(label.get_color()) > SECONDARY_TEXT_LUMINANCE:
            label.set_color(SECONDARY_TEXT_COLOR)
    ax.tick_params(axis="x", labelsize=AXIS_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=FEATURE_LABEL_SIZE)
    ax.set_xlabel("Model output (log-odds)", fontsize=FEATURE_LABEL_SIZE, labelpad=12.0)

    # "E[f(X)] = v" and "f(x) = v" are each a pair of tick labels that shap butts together
    # with left-aligned nudges sized for the width of 12pt mathtext. Those widths differ
    # per renderer, so LaTeX overlaps the pair. Anchoring the symbol's right edge and the
    # value's left edge to the tick instead keeps the gap fixed in every backend.
    for axis, shap_offsets in ((base_axis, (-20.0, 22.0)), (prediction_axis, (-10.0, 12.0))):
        labels = axis.xaxis.get_majorticklabels()
        alignments = ("right", "left")
        wanted_offsets = (-1.0, 1.0)
        for label, shap_offset, alignment, wanted in zip(
                labels, shap_offsets, alignments, wanted_offsets):
            label.set_fontsize(ENDPOINT_LABEL_SIZE)
            label.set_horizontalalignment(alignment)
            ascii_minus(label)
            label.set_transform(label.get_transform() + ScaledTranslation(
                (wanted - shap_offset) / 72.0, 0, fig.dpi_scale_trans))
        labels[1].set_color(SECONDARY_TEXT_COLOR)

    ax.set_title(title, fontsize=TITLE_SIZE, pad=6.0)
    fit_endpoint_labels(fig, ax, (base_axis, prediction_axis))
    reflow_overflowing_bar_labels(fig, ax)


def fit_endpoint_labels(fig, ax, annotation_axes):
    """Widen the x range until the f(x)/E[f(X)] callouts fit inside the canvas.

    They hang off whichever end of the waterfall the prediction lands on, so the amount
    of room needed differs per sample and cannot be a fixed margin.
    """
    for _ in range(3):
        fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        to_data = ax.transData.inverted()
        low, high = ax.get_xlim()
        for annotation_axis in annotation_axes:
            for label in annotation_axis.get_xticklabels():
                box = label.get_window_extent(renderer=renderer)
                low = min(low, to_data.transform((box.x0, 0))[0])
                high = max(high, to_data.transform((box.x1, 0))[0])
        margin = (high - low) * 0.015
        for axis in (ax, *annotation_axes):
            axis.set_xlim(low - margin, high + margin)


def reflow_overflowing_bar_labels(fig, ax):
    """Move white in-bar labels out whenever they no longer fit the bar.

    shap decides inside-vs-outside against its own 8in canvas, and each bar tapers to an
    arrow head, so a label that just fit there can end up printed over the head where
    only its middle band is still on colour. Re-deciding here against the real geometry.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted()
    arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrow)]
    point_in_data = (to_data.transform((fig.dpi / 72.0, 0))[0] - to_data.transform((0, 0))[0])

    for text in ax.texts:
        if to_rgb(text.get_color()) != (1.0, 1.0, 1.0):
            continue
        text_box = text.get_window_extent(renderer=renderer)
        arrow = next((a for a in arrows
                      if a.get_window_extent(renderer=renderer).overlaps(text_box)), None)
        if arrow is None:
            continue
        corners = sorted({round(float(x), 9) for x, _ in arrow.get_xy()})
        positive = not text.get_text().lstrip().startswith("-")
        tip, head_base = (corners[-1], corners[-2]) if positive else (corners[0], corners[1])
        head = abs(tip - head_base)
        shaft = abs(corners[-1] - corners[0]) - 2.0 * head
        if text_box.width * 1.05 <= shaft / point_in_data:
            continue
        pad = 4.0 * point_in_data
        text.set_position((tip + pad if positive else tip - pad, text.get_position()[1]))
        text.set_horizontalalignment("left" if positive else "right")
        text.set_color(darken(arrow.get_facecolor(), OUTSIDE_LABEL_DARKENING))


def save_all_formats(fig, tag):
    for figure_format in FORMATS:
        rc_params = {
            "figure.dpi": EXPORT_DPI,
            "savefig.dpi": EXPORT_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
        if figure_format == "pgf":
            rc_params["pgf.texsystem"] = "pdflatex"
            rc_params["pgf.rcfonts"] = False
            rc_params["pgf.preamble"] = "\n".join(PGF_PREAMBLE)
        with matplotlib.rc_context(rc_params):
            for out in out_paths(tag, figure_format):
                os.makedirs(os.path.dirname(out), exist_ok=True)
                fig.savefig(out, format=figure_format, dpi=EXPORT_DPI)
                print("wrote %s" % out)


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

    plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})
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
        restyle_waterfall(fig, f"{title}  ($\\hat p$={proba[i]:.2f})")
        save_all_formats(fig, tag)
        plt.close(fig)


if __name__ == "__main__":
    main()
