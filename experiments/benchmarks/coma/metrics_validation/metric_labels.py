"""Single source of truth for the metric naming used in the paper figures.

The LaTeX section (``papers/coma/latex_draft/sec/3_task_definition.tex``) names each
score with a compact symbol: the characteristic abbreviation with the reduction as a
subscript (e.g. ``direction_nearest`` -> $\\mathrm{Dir}_\\mathrm{near}$). Every figure
must reuse these exact symbols so a plot label identifies its formula. Keeping the
mapping here ensures the beeswarm and the metric-overview agree.
"""

# code id -> compact mathtext symbol (matches Table~\ref{tab:metric-inventory})
LABELS = {
    "direction_nearest": r"$\mathrm{Dir}_\mathrm{near}$",
    "far_mean":          r"$\mathrm{FAR}_\mathrm{mean}$",
    "coverage_mean":     r"$\mathrm{Cov}_\mathrm{mean}$",
    "height_mean":       r"$\mathrm{Hgt}_\mathrm{mean}$",
    "circ_nearest":      r"$\mathrm{Circ}_\mathrm{near}$",
    "elong_mean":        r"$\mathrm{Elo}_\mathrm{mean}$",
    "courtyard_mean":    r"$\mathrm{Crt}_\mathrm{mean}$",
    "exdiv_nearest":     r"$\mathrm{ExD}_\mathrm{near}$",
    "eldiv_nearest":     r"$\mathrm{ElD}_\mathrm{near}$",
    "frechet_orient":    r"$\mathrm{Fr}_\mathrm{orient}$",
    "orient_kde":        r"$\mathrm{KDE}_\mathrm{orient}$",
    "frechet_classic":   r"$\mathrm{Fr}_\mathrm{multi}$",
    "direction_mean":    r"$\mathrm{Dir}_\mathrm{mean}$",
    "height_nearest":    r"$\mathrm{Hgt}_\mathrm{near}$",
    "setback_mean":      r"$\mathrm{Set}_\mathrm{mean}$",
    "stepback_mean":     r"$\mathrm{Stp}_\mathrm{mean}$",
    "anglediv_nearest":  r"$\mathrm{AgD}_\mathrm{near}$",
}

# code id -> family used for figure colouring (mirrors the characteristic families
# of Section 3.2.2: orientation / scale & density / shape, plus the distribution metrics)
GROUP = {
    "direction_nearest": "orientation",
    "direction_mean":    "orientation",
    "frechet_orient":    "distribution",
    "orient_kde":        "distribution",
    "frechet_classic":   "distribution",
    "far_mean":          "scale_density",
    "coverage_mean":     "scale_density",
    "height_mean":       "scale_density",
    "height_nearest":    "scale_density",
    "setback_mean":      "scale_density",
    "circ_nearest":      "shape",
    "elong_mean":        "shape",
    "courtyard_mean":    "shape",
    "exdiv_nearest":     "shape",
    "eldiv_nearest":     "shape",
    "stepback_mean":     "shape",
    "anglediv_nearest":  "shape",
}

GROUP_COLOR = {
    "ensemble":      "#C44E52",
    "orientation":   "#4C72B0",
    "distribution":  "#3C9C8E",
    "scale_density": "#DD8452",
    "shape":         "#8C8C8C",
}

GROUP_LABEL = {
    "ensemble":      "Learned ensemble",
    "orientation":   "Orientation",
    "distribution":  "Distribution",
    "scale_density": "Scale & density",
    "shape":         "Shape",
}
