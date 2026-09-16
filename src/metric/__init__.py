__all__ = [
    "Metric",
    "SafeIgnoreSampleMerging",
    "SampleMetricROCAUC",
    "SampleMetricF1",
]

_LAZY_IMPORTS = {
    "Metric": (".metric", "Metric"),
    "SafeIgnoreSampleMerging": (".safe_ignore_sample_merging", "SafeIgnoreSampleMerging"),
    "SampleMetricROCAUC": (".sample_metric_roc_auc", "SampleMetricROCAUC"),
    "SampleMetricF1": (".sample_metric_f1", "SampleMetricF1"),
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")