__all__ = [
    "ImageDataVisualizer",
    "FocusMeshVisualizer",
    "MapRegionVisualizer",
    "MultiBarplotDatasetVisualizer",
    "FeatureRangeDatasetMultiSamplesVisualizer",
    "ContourHeatmapGridDatasetVisualizer",
    "MultiLineGridDatasetVisualizer",
    "MassingContextSplitVisualizer",
    "RegionMetricMapVisualizer",
    "MassingSimilarityHeatmapVisualizer"
]

_LAZY_IMPORTS = {
    "ImageDataVisualizer": (".image_data_visualizer", "ImageDataVisualizer"),
    "FocusMeshVisualizer": (".focus_mesh_visualizer", "FocusMeshVisualizer"),
    "MapRegionVisualizer": (".map_region_visualizer", "MapRegionVisualizer"),
    "MultiBarplotDatasetVisualizer": (".multi_barplot_dataset_visualizer", "MultiBarplotDatasetVisualizer"),
    "FeatureRangeDatasetMultiSamplesVisualizer": (".feature_range_dataset_multi_samples_visualizer", "FeatureRangeDatasetMultiSamplesVisualizer"),
    "ContourHeatmapGridDatasetVisualizer": (".contour_heatmap_grid_dataset_visualizer", "ContourHeatmapGridDatasetVisualizer"),
    "MultiLineGridDatasetVisualizer": (".multi_line_grid_dataset_visualizer", "MultiLineGridDatasetVisualizer"),
    "MassingContextSplitVisualizer": (".massing_context_split_visualizer", "MassingContextSplitVisualizer"),
    "RegionMetricMapVisualizer": (".region_metric_map_visualizer", "RegionMetricMapVisualizer"),
    "MassingSimilarityHeatmapVisualizer": (".massing_similarity_heatmap_visualizer", "MassingSimilarityHeatmapVisualizer")
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
