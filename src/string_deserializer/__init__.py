__all__ = [
    "StringDeserializer",
    "RegexJsonDeserializer",
    "RegexArgsDeserializer"
]

_LAZY_IMPORTS = {
    "StringDeserializer": (".string_deserializer", "StringDeserializer"),
    "RegexJsonDeserializer": (".regex_json_deserializer", "RegexJsonDeserializer"),
    "RegexArgsDeserializer": (".regex_args_deserializer", "RegexArgsDeserializer")
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")