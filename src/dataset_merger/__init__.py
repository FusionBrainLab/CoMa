__all__ = [
    "DatasetMerger",
    "GeoPandasDatasetIntersector",
    "GeoPandasRadiusDatasetJoiner",
    "CoMaBuildingsRegionsMetadataMerger",
    "KeyDatasetJoiner",
]

_LAZY_IMPORTS = {
    "DatasetMerger": (".dataset_merger", "DatasetMerger"),
    "GeoPandasDatasetIntersector": (".geo_pandas_dataset_intersector", "GeoPandasDatasetIntersector"),
    "GeoPandasRadiusDatasetJoiner": (".geo_pandas_radius_dataset_joiner", "GeoPandasRadiusDatasetJoiner"),
    "CoMaBuildingsRegionsMetadataMerger": (
        ".coma_buildings_regions_metadata_merger",
        "CoMaBuildingsRegionsMetadataMerger",
    ),
    "KeyDatasetJoiner": (".key_dataset_joiner", "KeyDatasetJoiner"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
