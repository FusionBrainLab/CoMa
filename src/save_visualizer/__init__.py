__all__ = [
    "SaveVisualizer",
    "FocusMeshHtmlVisualizer",
    "KeplerglPolygonsHtmlVisualizer",
    "KeplerglMultiPolygonsHtmlVisualizer",
    "SaveImageVisualizer"
]

_LAZY_IMPORTS = {
    "SaveVisualizer": (".save_visualizer", "SaveVisualizer"),
    "FocusMeshHtmlVisualizer": (".focus_mesh_html_visualizer", "FocusMeshHtmlVisualizer"),
    "KeplerglPolygonsHtmlVisualizer": (".keplergl_polygons_html_visualizer", "KeplerglPolygonsHtmlVisualizer"),
    "KeplerglMultiPolygonsHtmlVisualizer": (".keplergl_multi_polygons_html_visualizer", "KeplerglMultiPolygonsHtmlVisualizer"),
    "SaveImageVisualizer": (".save_image_visualizer", "SaveImageVisualizer"),
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")