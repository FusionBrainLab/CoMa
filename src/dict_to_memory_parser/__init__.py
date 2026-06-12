__all__ = [
    "DictToMemoryParser",
    "ContentChainDTMParser",
    "ConcatDTMParser",
    "EmptyColAdaptiveDTMParser",
    "MessagesDTMParser",
    "MultiColContentChainDTMParser"
]

_LAZY_IMPORTS = {
    "DictToMemoryParser": (".dict_to_memory_parser", "DictToMemoryParser"),
    "ContentChainDTMParser": (".content_chain_dtm_parser", "ContentChainDTMParser"),
    "ConcatDTMParser": (".concat_dtm_parser", "ConcatDTMParser"),
    "EmptyColAdaptiveDTMParser": (".empty_col_adaptive_dtm_parser", "EmptyColAdaptiveDTMParser"),
    "MessagesDTMParser": (".messages_dtm_parser", "MessagesDTMParser"),
    "MultiColContentChainDTMParser": (".multi_col_content_chain_dtm_parser", "MultiColContentChainDTMParser")
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")