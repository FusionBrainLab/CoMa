__all__ = [
    "HFProcessorCreator",
    "PretrainedProcessorCreator",
]

_LAZY_IMPORTS = {
    "HFProcessorCreator": (".hf_processor_creator", "HFProcessorCreator"),
    "PretrainedProcessorCreator": (".pretrained_processor_creator", "PretrainedProcessorCreator"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
