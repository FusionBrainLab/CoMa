__all__ = [
    "TensorBatchProcessor",
    "LabelsExtractor",
    "CompositionTensorProcessor",
]

_LAZY_IMPORTS = {
    "TensorBatchProcessor": (".tensor_batch_processor", "TensorBatchProcessor"),
    "LabelsExtractor": (".labels_extractor", "LabelsExtractor"),
    "CompositionTensorProcessor": (".composition_tensor_processor", "CompositionTensorProcessor"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
