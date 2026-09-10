import tempfile
import unittest

import numpy as np

from src.metric_validation.feature_sets import (
    ORIENTATION_RELATED_FEATURES,
    resolve_feature_set,
)
from src.sample_metric.learned_ensemble_metric import (
    LearnedEnsembleMetric,
    save_ensemble_artifact,
)


ALL_FEATURES = [
    "direction_nearest",
    "far_mean",
    "coverage_mean",
    "height_mean",
    "circ_nearest",
    "elong_mean",
    "courtyard_mean",
    "exdiv_nearest",
    "eldiv_nearest",
    "frechet_orient",
    "orient_kde",
    "frechet_classic",
    "direction_mean",
    "height_nearest",
    "setback_mean",
    "stepback_mean",
    "anglediv_nearest",
]


class ConstantEstimator:
    def predict_proba(self, x):
        return np.column_stack([np.zeros(len(x)), np.ones(len(x))])


class CapturingMetric:
    def __init__(self):
        self.seen_massing = None

    def __call__(self, *, sample):
        self.seen_massing = sample["massing"]
        return 0.5


class NoOrientationAblationTest(unittest.TestCase):
    def test_named_feature_set_excludes_orientation_related_features(self):
        resolved = resolve_feature_set("no_orientation", ALL_FEATURES)

        self.assertEqual(len(resolved), len(ALL_FEATURES) - len(ORIENTATION_RELATED_FEATURES))
        self.assertTrue(ORIENTATION_RELATED_FEATURES.isdisjoint(resolved))
        self.assertIn("anglediv_nearest", resolved)

    def test_unknown_string_feature_set_fails_loudly(self):
        with self.assertRaises(ValueError):
            resolve_feature_set("not_a_feature_set", ALL_FEATURES)

    def test_learned_metric_can_score_generated_massing_columns(self):
        metric = CapturingMetric()
        with tempfile.NamedTemporaryFile(suffix=".joblib") as artifact:
            save_ensemble_artifact(
                artifact.name,
                estimator=ConstantEstimator(),
                feature_names=["dummy"],
                standardize=False,
                mean=None,
                std=None,
            )
            ensemble = LearnedEnsembleMetric(
                model_path=artifact.name,
                feature_metrics={"dummy": metric},
                sample_massing_key="pred_massing",
            )

            score = ensemble(sample={"massing": "ground_truth", "pred_massing": "generated"})

        self.assertEqual(score, 1.0)
        self.assertEqual(metric.seen_massing, "generated")


if __name__ == "__main__":
    unittest.main()
