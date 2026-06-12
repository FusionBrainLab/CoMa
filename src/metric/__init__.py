"""from .metric import *
from .filter_length import *
from .sample_merging import *
from .safe_preprocess_sample_merging import *
from .agent_indicator_accuracy import *
from .filter_metric import *
from .safe_ignore_sample_merging import *
from .safe_preprocess_ignore_sample_merging import *
from .sample_metric_roc_auc import *"""

__all__ = [
    "Metric",
    "FilterLength",
    "SampleMerging",
    "SafePreprocessSampleMerging",
    "AgentIndicatorAccuracy",
    "FilterMetric",
    "SafeIgnoreSampleMerging",
    "SafePreprocessIgnoreSampleMerging",
    "SampleMetricROCAUC",
]

_LAZY_IMPORTS = {
    "Metric": (".metric", "Metric"),
    "FilterLength": (".filter_length", "FilterLength"),
    "SampleMerging": (".sample_merging", "SampleMerging"),
    "SafePreprocessSampleMerging": (".safe_preprocess_sample_merging", "SafePreprocessSampleMerging"),
    "AgentIndicatorAccuracy": (".agent_indicator_accuracy", "AgentIndicatorAccuracy"),
    "FilterMetric": (".filter_metric", "FilterMetric"),
    "SafeIgnoreSampleMerging": (".safe_ignore_sample_merging", "SafeIgnoreSampleMerging"),
    "SafePreprocessIgnoreSampleMerging": (".safe_preprocess_ignore_sample_merging", "SafePreprocessIgnoreSampleMerging"),
    "SampleMetricROCAUC": (".sample_metric_roc_auc", "SampleMetricROCAUC"),
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")