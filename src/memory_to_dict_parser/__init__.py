__all__ = [
    "MemoryToDictParser",
    "ProcessorVLLMInputCreator",
]

_LAZY_IMPORTS = {
    "MemoryToDictParser": (".memory_to_dict_parser", "MemoryToDictParser"),
    "ProcessorVLLMInputCreator": (".processor_vllm_input_creator", "ProcessorVLLMInputCreator"),
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")