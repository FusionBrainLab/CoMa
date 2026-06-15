"""Cross-validation / threshold helpers shared by the metric-validation experiments.

Plain module-level functions (not ``Function`` classes) so they stay lightweight and
are reused by both the learned-ensemble and feature-analysis experiments. The model
is always injected as ``make_clf`` — a zero-arg callable returning a *fresh*
estimator (e.g. ``lambda: clone(estimator)``) so every CV fold trains independently.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def optimal_f1(y_true, scores):
    """Best F1 over all score thresholds, with its threshold.

    Vectorised O(n log n): sort by score, sweep the ``scores >= thr`` decision via
    cumulative TP, evaluating only at distinct-value boundaries (exact ``>=``
    semantics). ``F1 = 2·TP / (k + P)`` with ``k = TP + FP`` predicted positives.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    y_true, scores = y_true[finite], scores[finite]
    if len(scores) == 0 or y_true.sum() == 0:
        return 0.0, 0.5
    order = np.argsort(scores, kind="mergesort")[::-1]
    s, yt = scores[order], y_true[order]
    tp = np.cumsum(yt)
    k = np.arange(1, len(yt) + 1)
    P = int(y_true.sum())
    f1 = 2.0 * tp / (k + P)
    boundary = np.ones(len(s), dtype=bool)
    boundary[:-1] = s[:-1] != s[1:]
    f1 = np.where(boundary, f1, -1.0)
    best = int(np.argmax(f1))
    return float(f1[best]), float(s[best])


def grouped_cv(X, y, groups, make_clf, standardize=False, return_oof=False, n_splits=5):
    """GroupKFold CV → dict(auc, auc_std, f1, f1_std); optional out-of-fold probs."""
    aucs, f1s = [], []
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        Xtr, Xte = X[tr], X[te]
        if standardize:
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        clf = make_clf()
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)[:, 1]
        oof[te] = proba
        aucs.append(roc_auc_score(y[te], proba))
        f1s.append(optimal_f1(y[te], proba)[0])
    out = {
        "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
    }
    if return_oof:
        return out, oof
    return out


def cv_auc(X, y, groups, make_clf, standardize=False, n_splits=5):
    res = grouped_cv(X, y, groups, make_clf, standardize, n_splits=n_splits)
    return res["auc"], res["auc_std"]


def forward_selection(X, y, groups, names, make_clf, stop_delta=0.0, min_keep=4, n_splits=5):
    """Greedy forward feature selection by GroupKFold AUC."""
    remaining = list(range(len(names)))
    selected, steps, best = [], [], 0.0
    while remaining:
        best_fi, best_m, best_s = None, -1.0, 0.0
        for fi in remaining:
            m, s = cv_auc(X[:, selected + [fi]], y, groups, make_clf, n_splits=n_splits)
            if m > best_m:
                best_fi, best_m, best_s = fi, m, s
        selected.append(best_fi)
        remaining.remove(best_fi)
        delta = best_m - best
        best = best_m
        steps.append({
            "step": len(selected), "added": names[best_fi],
            "selected": [names[i] for i in selected],
            "auc": best_m, "auc_std": best_s, "delta": delta,
        })
        if delta < stop_delta and len(selected) > min_keep:
            break
    return steps


def permutation_importance(X, y, groups, names, make_clf, seed=42, n_splits=5):
    """AUC drop per feature when shuffled; one fit per fold, reused across features."""
    out = {fname: [] for fname in names}
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        auc0 = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
        rng = np.random.default_rng(seed)
        for fi, fname in enumerate(names):
            Xp = X[te].copy()
            Xp[:, fi] = rng.permutation(Xp[:, fi])
            auc1 = roc_auc_score(y[te], clf.predict_proba(Xp)[:, 1])
            out[fname].append(auc0 - auc1)
    return {fname: {"mean_drop": float(np.mean(d)), "std_drop": float(np.std(d))}
            for fname, d in out.items()}
