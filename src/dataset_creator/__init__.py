__all__ = [
    "DatasetCreator",
    "PreprocessDatasetCreator",
    "CsvDatasetCreator",
    "ConcatDatasetCreator",
    "InversedJsonDatasetLoader",
    "PartialDatasetLoader",
    "InversedJsonChunkDatasetLoader",
    "InversedJsonChunkPreprocessDatasetLoader",
    "MemoryDatasetLoader",
    "OSMBuildingsCreator",
    "OSMRegionsCreator",
    "CoMaRegionsCreator",
    "CoMaBuildingsCreator",
    "ValidationResultsLoader"
]

_LAZY_IMPORTS = {
    "DatasetCreator": (".dataset_creator", "DatasetCreator"),
    "PreprocessDatasetCreator": (".preprocess_dataset_creator", "PreprocessDatasetCreator"),
    "CsvDatasetCreator": (".csv_dataset_creator", "CsvDatasetCreator"),
    "ConcatDatasetCreator": (".concat_dataset_creator", "ConcatDatasetCreator"),
    "InversedJsonDatasetLoader": (".inversed_json_dataset_loader", "InversedJsonDatasetLoader"),
    "PartialDatasetLoader": (".partial_dataset_loader", "PartialDatasetLoader"),
    "InversedJsonChunkDatasetLoader": (".inversed_json_chunk_dataset_loader", "InversedJsonChunkDatasetLoader"),
    "InversedJsonChunkPreprocessDatasetLoader": (".inversed_json_chunk_preprocess_dataset_loader", "InversedJsonChunkPreprocessDatasetLoader"),
    "MemoryDatasetLoader": (".memory_dataset_loader", "MemoryDatasetLoader"),
    "OSMBuildingsCreator": (".osm_buildings_creator", "OSMBuildingsCreator"),
    "OSMRegionsCreator": (".osm_regions_creator", "OSMRegionsCreator"),
    "CoMaRegionsCreator": (".coma_regions_creator", "CoMaRegionsCreator"),
    "CoMaBuildingsCreator": (".coma_buildings_creator", "CoMaBuildingsCreator"),
    "ValidationResultsLoader": (".validation_results_loader", "ValidationResultsLoader")
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")