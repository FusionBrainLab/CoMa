"""from .sample_metric import *
from .pattern_match_indicator import *
from .valid_json_indicator import *
from .corners_softness import *
from .massing_footprint_bottleneck_rate import *
from .massing_buildings_metric import *
from .building_volume_continuity import *
from .building_footprint_continuity import *
from .building_extrusions_continuity import *
from .massing_buildings_intersection_rate import *
from .building_levels_self_intersection_rate import *
from .building_covered_extrusions_rate import *
from .building_extrusions_intersection_rate import *
from .building_zero_sides_rate import *
from .building_zero_heights_rate import *
from .building_elevations_order_validity import *
from .reverse_metric import *
from .bounded_sample_metric import *
from .feature_match import *
from .building_floor_count import *
from .massing_requirements_feature_match import *
from .building_total_area import *
from .id_iou import *
from .site_match import *
from .feature_comparison import *
from .massing_elements_counter import *
from .massing_side_length_diversity import *
from .massing_angles_diversity import *
from .massing_elevations_diversity import *
from .massing_extrusion_height_diversity import *
from .massing_footprints_metric import *
from .massing_polygons_metric import *
from .bounded_normalized_sample_metric import *
from .safe_sample_metric import *
from .requirements_numerical_feature import *
from .requirements_id_count import *
from .site_polygon_metric import *
from .site_requirements_area_complexity import *
from .massing_contextual_relevance import *
from .massing_floor_area_ratio import *
from .massing_coverage import *
from .massing_setback import *
from .massing_elongation import *
from .massing_courtyard_ratio import *
from .massing_stepback_ratio import *"""

__all__ = [
    "SampleMetric",
    "PatternMatchIndicator",
    "ValidJsonIndicator",
    "CornersSoftness",
    "MassingFootprintBottleneckRate",
    "MassingBuildingsMetric",
    "BuildingVolumeContinuity",
    "BuildingFootprintContinuity",
    "BuildingExtrusionsContinuity",
    "MassingBuildingsIntersectionRate",
    "BuildingLevelsSelfIntersectionRate",
    "BuildingCoveredExtrusionsRate",
    "BuildingExtrusionsIntersectionRate",
    "BuildingZeroSidesRate",
    "BuildingZeroHeightsRate",
    "BuildingElevationsOrderValidity",
    "ReverseMetric",
    "BoundedSampleMetric",
    "FeatureMatch",
    "BuildingFloorCount",
    "MassingRequirementsFeatureMatch",
    "BuildingTotalArea",
    "IdIoU",
    "SiteMatch",
    "FeatureComparison",
    "MassingElementsCounter",
    "MassingSideLengthDiversity",
    "MassingAnglesDiversity",
    "MassingElevationsDiversity",
    "MassingExtrusionHeightDiversity",
    "MassingFootprintsMetric",
    "MassingPolygonsMetric",
    "BoundedNormalizedSampleMetric",
    "SafeSampleMetric",
    "RequirementsNumericalFeature",
    "RequirementsIdCount",
    "SitePolygonMetric",
    "SiteRequirementsAreaComplexity",
    "MassingContextualRelevance",
    "MassingFloorAreaRatio",
    "MassingCoverage",
    "MassingSetback",
    "MassingElongation",
    "MassingCourtyardRatio",
    "MassingStepbackRatio",
]

_LAZY_IMPORTS = {
    "SampleMetric": (".sample_metric", "SampleMetric"),
    "PatternMatchIndicator": (".pattern_match_indicator", "PatternMatchIndicator"),
    "ValidJsonIndicator": (".valid_json_indicator", "ValidJsonIndicator"),
    "CornersSoftness": (".corners_softness", "CornersSoftness"),
    "MassingFootprintBottleneckRate": (".massing_footprint_bottleneck_rate", "MassingFootprintBottleneckRate"),
    "MassingBuildingsMetric": (".massing_buildings_metric", "MassingBuildingsMetric"),
    "BuildingVolumeContinuity": (".building_volume_continuity", "BuildingVolumeContinuity"),
    "BuildingFootprintContinuity": (".building_footprint_continuity", "BuildingFootprintContinuity"),
    "BuildingExtrusionsContinuity": (".building_extrusions_continuity", "BuildingExtrusionsContinuity"),
    "MassingBuildingsIntersectionRate": (".massing_buildings_intersection_rate", "MassingBuildingsIntersectionRate"),
    "BuildingLevelsSelfIntersectionRate": (".building_levels_self_intersection_rate", "BuildingLevelsSelfIntersectionRate"),
    "BuildingCoveredExtrusionsRate": (".building_covered_extrusions_rate", "BuildingCoveredExtrusionsRate"),
    "BuildingExtrusionsIntersectionRate": (".building_extrusions_intersection_rate", "BuildingExtrusionsIntersectionRate"),
    "BuildingZeroSidesRate": (".building_zero_sides_rate", "BuildingZeroSidesRate"),
    "BuildingZeroHeightsRate": (".building_zero_heights_rate", "BuildingZeroHeightsRate"),
    "BuildingElevationsOrderValidity": (".building_elevations_order_validity", "BuildingElevationsOrderValidity"),
    "ReverseMetric": (".reverse_metric", "ReverseMetric"),
    "BoundedSampleMetric": (".bounded_sample_metric", "BoundedSampleMetric"),
    "FeatureMatch": (".feature_match", "FeatureMatch"),
    "BuildingFloorCount": (".building_floor_count", "BuildingFloorCount"),
    "MassingRequirementsFeatureMatch": (".massing_requirements_feature_match", "MassingRequirementsFeatureMatch"),
    "BuildingTotalArea": (".building_total_area", "BuildingTotalArea"),
    "IdIoU": (".id_iou", "IdIoU"),
    "SiteMatch": (".site_match", "SiteMatch"),
    "FeatureComparison": (".feature_comparison", "FeatureComparison"),
    "MassingElementsCounter": (".massing_elements_counter", "MassingElementsCounter"),
    "MassingSideLengthDiversity": (".massing_side_length_diversity", "MassingSideLengthDiversity"),
    "MassingAnglesDiversity": (".massing_angles_diversity", "MassingAnglesDiversity"),
    "MassingElevationsDiversity": (".massing_elevations_diversity", "MassingElevationsDiversity"),
    "MassingExtrusionHeightDiversity": (".massing_extrusion_height_diversity", "MassingExtrusionHeightDiversity"),
    "MassingFootprintsMetric": (".massing_footprints_metric", "MassingFootprintsMetric"),
    "MassingPolygonsMetric": (".massing_polygons_metric", "MassingPolygonsMetric"),
    "BoundedNormalizedSampleMetric": (".bounded_normalized_sample_metric", "BoundedNormalizedSampleMetric"),
    "SafeSampleMetric": (".safe_sample_metric", "SafeSampleMetric"),
    "RequirementsNumericalFeature": (".requirements_numerical_feature", "RequirementsNumericalFeature"),
    "RequirementsIdCount": (".requirements_id_count", "RequirementsIdCount"),
    "SitePolygonMetric": (".site_polygon_metric", "SitePolygonMetric"),
    "SiteRequirementsAreaComplexity": (".site_requirements_area_complexity", "SiteRequirementsAreaComplexity"),
    "MassingContextualRelevance": (".massing_contextual_relevance", "MassingContextualRelevance"),
    "MassingFloorAreaRatio": (".massing_floor_area_ratio", "MassingFloorAreaRatio"),
    "MassingCoverage": (".massing_coverage", "MassingCoverage"),
    "MassingSetback": (".massing_setback", "MassingSetback"),
    "MassingElongation": (".massing_elongation", "MassingElongation"),
    "MassingCourtyardRatio": (".massing_courtyard_ratio", "MassingCourtyardRatio"),
    "MassingStepbackRatio": (".massing_stepback_ratio", "MassingStepbackRatio"),
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
