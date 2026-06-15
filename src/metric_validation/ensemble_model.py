"""Config-friendly wrapper pairing an sklearn-style estimator with a standardize flag.

Plain dataclass (not a ``Function``) so hydra can instantiate it with a nested
``estimator`` ``_target_`` (LogisticRegression / HistGradientBoostingClassifier /
CatBoostClassifier) and a ``standardize`` switch (linear models want it, trees do not).
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class EnsembleModel:
    estimator: Any
    standardize: bool = False
