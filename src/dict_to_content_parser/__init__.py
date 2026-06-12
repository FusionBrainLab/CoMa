__all__ = [
    "DictToContentParser",
    "ModalityKeyDTCParser",
    "StringFormatDTCParser"
]

_LAZY_IMPORTS = {
    "DictToContentParser": (".dict_to_content_parser", "DictToContentParser"),
    "ModalityKeyDTCParser": (".modality_key_dtc_parser", "ModalityKeyDTCParser"),
    "StringFormatDTCParser": (".string_format_dtc_parser", "StringFormatDTCParser")
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")