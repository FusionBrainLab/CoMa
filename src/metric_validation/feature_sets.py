"""Named feature-set definitions for learned metric ablations."""

from collections.abc import Iterable
from typing import Any, Sequence


ORIENTATION_RELATED_FEATURES = frozenset({
    "direction_nearest",
    "direction_mean",
    "frechet_orient",
    "orient_kde",
    "frechet_classic",
})

FEATURE_SET_ALIASES = {
    "no_orientation": lambda names: [
        name for name in names if name not in ORIENTATION_RELATED_FEATURES
    ],
}


def resolve_feature_set(feature_set: Any, available_names: Sequence[str]) -> list[str]:
    """Resolve config-friendly feature set names against an available score matrix."""
    names = list(available_names)
    if feature_set is None or feature_set == "all":
        return names
    if isinstance(feature_set, str):
        if feature_set in FEATURE_SET_ALIASES:
            return FEATURE_SET_ALIASES[feature_set](names)
        raise ValueError(
            "Unknown feature_set '%s'. Use 'all', one of %s, or an explicit list."
            % (feature_set, sorted(FEATURE_SET_ALIASES))
        )
    if not isinstance(feature_set, Iterable):
        raise TypeError("feature_set must be null, a string alias, or an iterable")
    return [feature for feature in feature_set if feature in names]
