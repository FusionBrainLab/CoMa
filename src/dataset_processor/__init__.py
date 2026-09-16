__all__ = [
    "DatasetProcessor",
    "CompositionProcessor",
    "FunctionProcessor",
    "JsonStringDumper",
    "JsonStringLoader",
    "ColumnDrop",
    "ShuffleProcessor",
    "GeoJSONToShapelyProcessor",
    "GeoProjectProcessor",
    "FootprintsToBuildingProcessor",
    "MassingContextSampler",
    "MassingContextConverter",
    "ModalityEstimationTokensCountFilter",
    "MassingAbsoluteCoordinatesProcessor",
    "SqueezeFeatureCreator",
    "UnsqueezeFeatureCreator",
    "MetricFeatureComputeProcessor",
    "IdRowListMerger",
    "MassingContextSimilarityComputer",
    "IdRowMeanMerger",
    "RegexColumnFilter",
    "ColumnLeft"
]

_LAZY_IMPORTS = {
    "DatasetProcessor": (".dataset_processor", "DatasetProcessor"),
    "CompositionProcessor": (".composition_processor", "CompositionProcessor"),
    "FunctionProcessor": (".function_processor", "FunctionProcessor"),
    "JsonStringDumper": (".json_string_dumper", "JsonStringDumper"),
    "JsonStringLoader": (".json_string_loader", "JsonStringLoader"),
    "ColumnDrop": (".column_drop", "ColumnDrop"),
    "ShuffleProcessor": (".shuffle_processor", "ShuffleProcessor"),
    "GeoJSONToShapelyProcessor": (".geojson_to_shapely_processor", "GeoJSONToShapelyProcessor"),
    "GeoProjectProcessor": (".geo_project_processor", "GeoProjectProcessor"),
    "FootprintsToBuildingProcessor": (".footprints_to_building_processor", "FootprintsToBuildingProcessor"),
    "MassingContextSampler": (".massing_context_sampler", "MassingContextSampler"),
    "MassingContextConverter": (".massing_context_converter", "MassingContextConverter"),
    "ModalityEstimationTokensCountFilter": (".modality_estimation_tokens_count_filter", "ModalityEstimationTokensCountFilter"),
    "MassingAbsoluteCoordinatesProcessor": (".massing_absolute_coordinates_processor", "MassingAbsoluteCoordinatesProcessor"),
    "SqueezeFeatureCreator": (".squeeze_feature_creator", "SqueezeFeatureCreator"),
    "UnsqueezeFeatureCreator": (".unsqueeze_feature_creator", "UnsqueezeFeatureCreator"),
    "MetricFeatureComputeProcessor": (".metric_feature_compute_processor", "MetricFeatureComputeProcessor"),
    "IdRowListMerger": (".id_row_list_merger", "IdRowListMerger"),
    "MassingContextSimilarityComputer": (".massing_context_similarity_computer", "MassingContextSimilarityComputer"),
    "IdRowMeanMerger": (".id_row_mean_merger", "IdRowMeanMerger"),
    "RegexColumnFilter": (".regex_column_filter", "RegexColumnFilter"),
    "ColumnLeft": (".column_left", "ColumnLeft"),
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")