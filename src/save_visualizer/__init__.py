__all__ = [
    "SaveVisualizer",
    "SaveImageVisualizer",
    "SaveFigureVisualizer"
]

_LAZY_IMPORTS = {
    "SaveVisualizer": (".save_visualizer", "SaveVisualizer"),
    "SaveImageVisualizer": (".save_image_visualizer", "SaveImageVisualizer"),
    "SaveFigureVisualizer": (".save_figure_visualizer", "SaveFigureVisualizer"),
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")