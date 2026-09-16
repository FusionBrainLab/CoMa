__all__ = [
    "ModelHandler",
    "HFModelSaver",
]

_LAZY_IMPORTS = {
    "ModelHandler": (".model_handler", "ModelHandler"),
    "HFModelSaver": (".hf_model_saver", "HFModelSaver"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
