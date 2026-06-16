"""Metric-validation experiment components (config-driven ``Function`` classes).

These orchestrate the contextual-relevance metric study (feature scoring, learned
ensembles incl. CatBoost, feature analysis, figures) on top of the existing
``SampleMetric`` / ``DatasetCreator`` building blocks, in the repo's hydra flow.
"""

__all__ = [
    "FeatureScoreMatrix",
    "EnsembleModel",
    "LearnedEnsembleExperiment",
    "LearnedEnsembleTrainer",
    "FeatureAnalysisExperiment",
    "MetricFiguresRenderer",
    "MetricsValidationExperiment",
]

_LAZY_IMPORTS = {
    "FeatureScoreMatrix": (".feature_score_matrix", "FeatureScoreMatrix"),
    "EnsembleModel": (".ensemble_model", "EnsembleModel"),
    "LearnedEnsembleExperiment": (".learned_ensemble_experiment", "LearnedEnsembleExperiment"),
    "LearnedEnsembleTrainer": (".learned_ensemble_trainer", "LearnedEnsembleTrainer"),
    "FeatureAnalysisExperiment": (".feature_analysis_experiment", "FeatureAnalysisExperiment"),
    "MetricFiguresRenderer": (".metric_figures_renderer", "MetricFiguresRenderer"),
    "MetricsValidationExperiment": (".metrics_validation_experiment", "MetricsValidationExperiment"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
