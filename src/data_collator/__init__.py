__all__ = [
    "DataCollator",
    "MemoryTokenizerCollator",
    "PostprocessCollator",
]

_LAZY_IMPORTS = {
    "DataCollator": (".data_collator", "DataCollator"),
    "MemoryTokenizerCollator": (".memory_tokenizer_collator", "MemoryTokenizerCollator"),
    "PostprocessCollator": (".postprocess_collator", "PostprocessCollator"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
