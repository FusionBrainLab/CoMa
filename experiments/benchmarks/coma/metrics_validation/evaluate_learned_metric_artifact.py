"""Evaluate a learned-metric artifact against a cached score matrix."""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(HERE, "models", "learned_ensemble_catboost_no_orientation.joblib")
DEFAULT_SCORES_PATH = os.path.join(HERE, "results", "scores.npz")
DEFAULT_OUTPUT_PATH = os.path.join(HERE, "results", "learned_metric_no_orientation_validation.json")


def evaluate(*, model_path: str, scores_path: str) -> dict:
    artifact = joblib.load(model_path)
    feature_names = list(artifact["feature_names"])
    scores = np.load(scores_path, allow_pickle=True)
    matrix_names = [str(name) for name in scores["names"]]
    columns = [matrix_names.index(name) for name in feature_names]
    x = scores["X"][:, columns].astype(float)
    y = scores["y"].astype(int)

    if artifact["standardize"] and artifact["mean"] is not None:
        x = (x - np.asarray(artifact["mean"], dtype=float)) / np.asarray(artifact["std"], dtype=float)

    pred = artifact["estimator"].predict_proba(x)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, pred)
    denom = precision[:-1] + recall[:-1]
    f1_scores = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    best_index = int(np.argmax(f1_scores))
    best = {
        "f1": float(f1_scores[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "threshold": float(thresholds[best_index]),
    }

    return {
        "model_path": model_path,
        "scores_path": scores_path,
        "n_samples": int(len(y)),
        "feature_names": feature_names,
        "roc_auc": float(roc_auc_score(y, pred)),
        **best,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scores-path", default=DEFAULT_SCORES_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    result = evaluate(model_path=args.model_path, scores_path=args.scores_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
