__all__ = [
    "MemoryTokenizer",
    "HFProcessor",
    "ParameterConfig",
    "DeepSeekOCR2Tokenizer",
    "ModalityFilterTokenizer"
]

_LAZY_IMPORTS = {
    "MemoryTokenizer": (".memory_tokenizer", "MemoryTokenizer"),
    "HFProcessor": (".hf_processor", "HFProcessor"),
    "ParameterConfig": (".hf_processor", "ParameterConfig"),
    "DeepSeekOCR2Tokenizer": (".deepseek_ocr_2_tokenizer", "DeepSeekOCR2Tokenizer"),
    "ModalityFilterTokenizer": (".modality_filter_tokenizer", "ModalityFilterTokenizer")
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")