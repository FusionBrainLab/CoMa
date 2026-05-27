from typing import List, Any, Dict, Literal

import pandas as pd
from tqdm import tqdm

from .dataset_creator import DatasetCreator
from ..interpretable_function import InterpretableFunction
from ..core.function_utils import FunctionWrapper, FunctionGraph
from ..core.parsers import (
    DictToDictParser, AnyToDictParser, DictToAnyParser
)
from ..dataset_creator import (
    CsvDatasetCreator,
    PreprocessDatasetCreator
)
from ..dataset_processor import (
    GeoJSONToShapelyProcessor,
    GeoProjectProcessor,
    CompositionProcessor,
    FunctionProcessor
)
from ..dataset_merger import GeoPandasDatasetIntersector
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter

class CoMaRegionsCreator(DatasetCreator):
    def __init__(self, *, properties_dataset_path: str,
                        id_col: str,
                        output_col: str) -> None:
        self.properties_dataset_path = properties_dataset_path
        self.id_col = id_col
        self.output_col = output_col
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()

    def __call__(self) -> Dict[str, List[Any]]:
        tqdm.pandas()
        #----------LOAD PROPERTIES----------
        loader = PreprocessDatasetCreator(
            base_creator=CsvDatasetCreator(
                path=self.properties_dataset_path,
                pandas_args={
                    "sep":",",
                    "index_col":False
                }
            ),
            preprocessor=CompositionProcessor(
                base_processors=[
                    GeoJSONToShapelyProcessor(
                        geojson_col="Geo Shape",
                        on_invalid="warn",
                        output_col="global_projection"
                    ),
                    GeoProjectProcessor(
                        geo_col="global_projection",
                        from_format="EPSG:4326",
                        to_format="EPSG:3857"
                    )
                ]
            )
        )
        properties = loader()

        properties = pd.DataFrame(properties)
        properties["id"] = [str(i) for i in range(len(properties))]
        properties["id_buffer"] = properties["id"]
        properties = properties.set_index("id_buffer")

        #----------INTERSECT PROPERTIES----------
        intersector = GeoPandasDatasetIntersector(
            intersect_dataset_key="intersect_dataset",
            intersect_dataset_geo_col="global_projection",
            intersect_dataset_id_col="id",
            main_dataset_key="main_dataset",
            main_dataset_geo_col="global_projection",
            main_dataset_id_col="id",
            drop_self_intersections=True,
            output_properties=["id", "area"],
            output_col="intersections"
        )
        datasets = {
            "intersect_dataset": properties.to_dict("list"),
            "main_dataset": properties.to_dict("list")
        }
        properties = intersector(datasets=datasets)
        properties = pd.DataFrame(properties)
        properties["id_buffer"] = properties["id"]
        properties = properties.set_index("id_buffer")

        #----------GET REGIONS----------
        def is_parent_property(row):
            intersections = row["intersections"]
            area = row["global_projection"].area
            is_top = True
            for intersection in intersections:
                local_area = properties.loc[intersection["id"]]["global_projection"].area
                intersection_area = intersection["area"]
                if intersection_area/area > intersection_area/local_area:
                    is_top = False
                    break
            return is_top
        regions = properties[properties.progress_apply(lambda row: is_parent_property(row), axis=1)]

        def get_property_ids(row):
            children = [spec["id"] for spec in row["intersections"] if spec["area"] > 0]
            children_properties = properties.loc[children]["Property_ID"].values.tolist()
            ids = [row["Property_ID"]] + children_properties
            return ids
        regions["properties"] = regions.progress_apply(lambda row: get_property_ids(row), axis=1)

        regions[self.output_col] = regions.progress_apply(lambda row: self.shapely_to_polygons_converter(polygons=row["global_projection"]), axis=1)
        regions = regions[["id", "properties", self.output_col]]
        return regions.to_dict("list")