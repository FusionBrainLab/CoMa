__all__ = [
    "DatasetProcessor",
    "OrderIdCreator",
    "CompositionProcessor",
    "IdRowMerger",
    "DatasetFilter",
    "FunctionProcessor",
    "JsonStringDumper",
    "SiteMassingRotationAugmenter",
    "ColumnMergeAugmenter",
    "RandomSampler",
    "SplitProcessor",
    "TokensCountFilter",
    "JsonStringLoader",
    "ErrorSafeFunctionProcessor",
    "ColumnDrop",
    "RelativePathsResolver",
    "SiteMassingShiftAugmenter",
    "ShuffleProcessor",
    "PointRadiusSelfJoiner",
    "GeoJSONToShapelyProcessor",
    "GeoProjectProcessor",
    "FootprintsToBuildingProcessor",
    "FeaturesUniformizeProcessor",
    "MassingContextSampler",
    "MassingContextConverter",
    "ModalityEstimationTokensCountFilter"
]

_LAZY_IMPORTS = {
    "DatasetProcessor": (".dataset_processor", "DatasetProcessor"),
    "OrderIdCreator": (".order_id_creator", "OrderIdCreator"),
    "CompositionProcessor": (".composition_processor", "CompositionProcessor"),
    "IdRowMerger": (".id_row_merger", "IdRowMerger"),
    "DatasetFilter": (".dataset_filter", "DatasetFilter"),
    "FunctionProcessor": (".function_processor", "FunctionProcessor"),
    "JsonStringDumper": (".json_string_dumper", "JsonStringDumper"),
    "SiteMassingRotationAugmenter": (".site_massing_rotation_augmenter", "SiteMassingRotationAugmenter"),
    "ColumnMergeAugmenter": (".column_merge_augmenter", "ColumnMergeAugmenter"),
    "RandomSampler": (".random_sampler", "RandomSampler"),
    "SplitProcessor": (".split_processor", "SplitProcessor"),
    "TokensCountFilter": (".tokens_count_filter", "TokensCountFilter"),
    "JsonStringLoader": (".json_string_loader", "JsonStringLoader"),
    "ErrorSafeFunctionProcessor": (".error_safe_function_processor", "ErrorSafeFunctionProcessor"),
    "ColumnDrop": (".column_drop", "ColumnDrop"),
    "RelativePathsResolver": (".relative_paths_resolver", "RelativePathsResolver"),
    "SiteMassingShiftAugmenter": (".site_massing_shift_augmenter", "SiteMassingShiftAugmenter"),
    "ShuffleProcessor": (".shuffle_processor", "ShuffleProcessor"),
    "PointRadiusSelfJoiner": (".point_radius_self_joiner", "PointRadiusSelfJoiner"),
    "GeoJSONToShapelyProcessor": (".geojson_to_shapely_processor", "GeoJSONToShapelyProcessor"),
    "GeoProjectProcessor": (".geo_project_processor", "GeoProjectProcessor"),
    "FootprintsToBuildingProcessor": (".footprints_to_building_processor", "FootprintsToBuildingProcessor"),
    "FeaturesUniformizeProcessor": (".features_uniformize_processor", "FeaturesUniformizeProcessor"),
    "MassingContextSampler": (".massing_context_sampler", "MassingContextSampler"),
    "MassingContextConverter": (".massing_context_converter", "MassingContextConverter"),
    "ModalityEstimationTokensCountFilter": (".modality_estimation_tokens_count_filter", "ModalityEstimationTokensCountFilter")
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")