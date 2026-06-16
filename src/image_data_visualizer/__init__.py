__all__ = [
    "ImageDataVisualizer",
    "ImageMassingVisualizer",
    "IsometricMassingMeshVisualizer",
    "ContextualMassingMeshVisualizer",
    "FocusMeshVisualizer",
    "SeabornBarplotCreator",
    "SeabornDistributionVisualizer",
    "ContextualMeshVisualizer",
    "FloorSeparatedMassingMeshVisualizer",
    "MapRegionVisualizer",
    "HeatmapDatasetVisualizer",
    "ContourHeatmapDatasetVisualizer",
    "MultiLineDatasetVisualizer",
    "MultiDatasetSamplesVisualizer",
    "MultiBarplotDatasetVisualizer"
]

_LAZY_IMPORTS = {
    "ImageDataVisualizer": (".image_data_visualizer", "ImageDataVisualizer"),
    "ImageMassingVisualizer": (".image_massing_visualizer", "ImageMassingVisualizer"),
    "IsometricMassingMeshVisualizer": (".isometric_massing_mesh_visualizer", "IsometricMassingMeshVisualizer"),
    "ContextualMassingMeshVisualizer": (".contextual_massing_mesh_visualizer", "ContextualMassingMeshVisualizer"),
    "FocusMeshVisualizer": (".focus_mesh_visualizer", "FocusMeshVisualizer"),
    "SeabornBarplotCreator": (".seaborn_barplot_creator", "SeabornBarplotCreator"),
    "SeabornDistributionVisualizer": (".seaborn_distribution_visualizer", "SeabornDistributionVisualizer"),
    "ContextualMeshVisualizer": (".contextual_mesh_visualizer", "ContextualMeshVisualizer"),
    "FloorSeparatedMassingMeshVisualizer": (".floor_separated_massing_mesh_visualizer", "FloorSeparatedMassingMeshVisualizer"),
    "MapRegionVisualizer": (".map_region_visualizer", "MapRegionVisualizer"),
    "HeatmapDatasetVisualizer": (".heatmap_dataset_visualizer", "HeatmapDatasetVisualizer"),
    "ContourHeatmapDatasetVisualizer": (".contour_heatmap_dataset_visualizer", "ContourHeatmapDatasetVisualizer"),
    "MultiLineDatasetVisualizer": (".multi_line_dataset_visualizer", "MultiLineDatasetVisualizer"),
    "MultiDatasetSamplesVisualizer": (".multi_dataset_samples_visualizer", "MultiDatasetSamplesVisualizer"),
    "MultiBarplotDatasetVisualizer": (".multi_barplot_dataset_visualizer", "MultiBarplotDatasetVisualizer")
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
