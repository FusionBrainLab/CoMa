__all__ = [
    "DatasetHandler",
    "PreprocessDatasetHandler",
    "InversedJsonChunkDatasetSaver",
    "InversedJsonChunkTreeDatasetSaver",
]

_LAZY_IMPORTS = {
    "DatasetHandler": (".dataset_handler", "DatasetHandler"),
    "PreprocessDatasetHandler": (".preprocess_dataset_handler", "PreprocessDatasetHandler"),
    "InversedJsonChunkDatasetSaver": (".inversed_json_chunk_dataset_saver", "InversedJsonChunkDatasetSaver"),
    "InversedJsonChunkTreeDatasetSaver": (
        ".inversed_json_chunk_tree_dataset_saver",
        "InversedJsonChunkTreeDatasetSaver",
    ),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
