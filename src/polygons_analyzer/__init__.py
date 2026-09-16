__all__ = [
    "PolygonsAnalyzer",
    "ConvexRatePA",
    "DescribedRectangleRatePA",
    "DescribedCircleRatePA",
    "InscribedCircleRatePA",
    "CircularityPA",
    "DirectionPA",
]

_LAZY_IMPORTS = {
    "PolygonsAnalyzer": (".polygons_analyzer", "PolygonsAnalyzer"),
    "ConvexRatePA": (".convex_rate_pa", "ConvexRatePA"),
    "DescribedRectangleRatePA": (".described_rectangle_rate_pa", "DescribedRectangleRatePA"),
    "DescribedCircleRatePA": (".described_circle_rate_pa", "DescribedCircleRatePA"),
    "InscribedCircleRatePA": (".inscribed_circle_rate_pa", "InscribedCircleRatePA"),
    "CircularityPA": (".circularity_pa", "CircularityPA"),
    "DirectionPA": (".direction_pa", "DirectionPA"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
