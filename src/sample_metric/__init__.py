__all__ = [
    "SampleMetric",
    "MassingBuildingsMetric",
    "BuildingFloorCount",
    "MassingElementsCounter",
    "MassingSideLengthDiversity",
    "MassingAnglesDiversity",
    "MassingElevationsDiversity",
    "MassingExtrusionHeightDiversity",
    "MassingFootprintsMetric",
    "MassingPolygonsMetric",
    "MassingContextualRelevance",
    "MassingFloorAreaRatio",
    "MassingCoverage",
    "MassingSetback",
    "MassingElongation",
    "MassingCourtyardRatio",
    "MassingStepbackRatio",
    "MultiFeatureContextualRelevance",
    "MassingContextFrechetDistance",
    "MassingOrientationHistMatch",
    "LearnedEnsembleMetric",
]

_LAZY_IMPORTS = {
    "SampleMetric": (".sample_metric", "SampleMetric"),
    "MassingBuildingsMetric": (".massing_buildings_metric", "MassingBuildingsMetric"),
    "BuildingFloorCount": (".building_floor_count", "BuildingFloorCount"),
    "MassingElementsCounter": (".massing_elements_counter", "MassingElementsCounter"),
    "MassingSideLengthDiversity": (".massing_side_length_diversity", "MassingSideLengthDiversity"),
    "MassingAnglesDiversity": (".massing_angles_diversity", "MassingAnglesDiversity"),
    "MassingElevationsDiversity": (".massing_elevations_diversity", "MassingElevationsDiversity"),
    "MassingExtrusionHeightDiversity": (".massing_extrusion_height_diversity", "MassingExtrusionHeightDiversity"),
    "MassingFootprintsMetric": (".massing_footprints_metric", "MassingFootprintsMetric"),
    "MassingPolygonsMetric": (".massing_polygons_metric", "MassingPolygonsMetric"),
    "MassingContextualRelevance": (".massing_contextual_relevance", "MassingContextualRelevance"),
    "MassingFloorAreaRatio": (".massing_floor_area_ratio", "MassingFloorAreaRatio"),
    "MassingCoverage": (".massing_coverage", "MassingCoverage"),
    "MassingSetback": (".massing_setback", "MassingSetback"),
    "MassingElongation": (".massing_elongation", "MassingElongation"),
    "MassingCourtyardRatio": (".massing_courtyard_ratio", "MassingCourtyardRatio"),
    "MassingStepbackRatio": (".massing_stepback_ratio", "MassingStepbackRatio"),
    "MultiFeatureContextualRelevance": (".multi_feature_contextual_relevance", "MultiFeatureContextualRelevance"),
    "MassingContextFrechetDistance": (".massing_context_frechet_distance", "MassingContextFrechetDistance"),
    "MassingOrientationHistMatch": (".massing_orientation_hist_match", "MassingOrientationHistMatch"),
    "LearnedEnsembleMetric": (".learned_ensemble_metric", "LearnedEnsembleMetric"),
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
