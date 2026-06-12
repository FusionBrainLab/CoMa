__all__ = [
    "ImagesParser",
    "MemoryToDictParser",
    "MessagesMerger",
    "TextMessagesParser",
    "ProcessorVLLMInputCreator",
    "ResizeImagesParser",
    "DictMessagesCreator"
]

_LAZY_IMPORTS = {
    "ImagesParser": (".images_parser", "ImagesParser"),
    "MemoryToDictParser": (".memory_to_dict_parser", "MemoryToDictParser"),
    "MessagesMerger": (".messages_merger", "MessagesMerger"),
    "TextMessagesParser": (".text_messages_parser", "TextMessagesParser"),
    "ProcessorVLLMInputCreator": (".processor_vllm_input_creator", "ProcessorVLLMInputCreator"),
    "ResizeImagesParser": (".resize_images_parser", "ResizeImagesParser"),
    "DictMessagesCreator": (".dict_messages_creator", "DictMessagesCreator")
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")