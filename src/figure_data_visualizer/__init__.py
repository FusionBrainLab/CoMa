__all__ = [
    "FigureDataVisualizer",
    "MultiBarplotFigureVisualizer",
    "MultiLineGridFigureVisualizer",
    "ContourHeatmapGridFigureVisualizer"
]

_LAZY_IMPORTS = {
    "FigureDataVisualizer": (".figure_data_visualizer", "FigureDataVisualizer"),
    "MultiBarplotFigureVisualizer": (".multi_barplot_figure_visualizer", "MultiBarplotFigureVisualizer"),
    "MultiLineGridFigureVisualizer": (".multi_line_grid_figure_visualizer", "MultiLineGridFigureVisualizer"),
    "ContourHeatmapGridFigureVisualizer": (".contour_heatmap_grid_figure_visualizer", "ContourHeatmapGridFigureVisualizer")
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
