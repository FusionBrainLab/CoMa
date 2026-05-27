from typing import List, Any, Dict, Literal
from datetime import datetime

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
    FunctionProcessor,
    FootprintsToBuildingProcessor
)
from ..dataset_merger import GeoPandasDatasetIntersector
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter

class CoMaBuildingsCreator(DatasetCreator):
    def __init__(self, *, footprints_dataset_path: str,
                        building_id_col: str,
                        time_col: str) -> None:
        self.footprints_dataset_path = footprints_dataset_path
        self.building_id_col = building_id_col
        self.time_col = time_col
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()

    def __call__(self) -> Dict[str, List[Any]]:
        tqdm.pandas()
        #----------LOAD FOOTPRINTS----------
        loader=CsvDatasetCreator(
            path=self.footprints_dataset_path,
            pandas_args={
                "sep":",",
                "index_col":False
            }
        )
        footprints = loader()
        footprints = pd.DataFrame(footprints)

        #----------GET NEWEST FOOTPRINTS----------
        id_to_dates = {}
        def get_dates(row):
            id = row[self.building_id_col]
            date = row[self.time_col]
            if id not in id_to_dates:
                id_to_dates[id] = []
            id_to_dates[id].append(date)
        footprints.progress_apply(lambda row: get_dates(row), axis=1)

        id_to_max_date = {id:max([datetime.strptime(str(d), "%Y%m%d") for d in dates]) for id, dates in id_to_dates.items()}
        footprints = footprints[footprints.progress_apply(lambda row: datetime.strptime(str(row[self.time_col]), "%Y%m%d") == id_to_max_date[row[self.building_id_col]], axis=1)]

        #----------CONVERT POLYGONS----------
        footprints = footprints.to_dict("list")
        shapely_converter = GeoJSONToShapelyProcessor(
            geojson_col="Geo Shape",
            on_invalid="warn",
            output_col="global_projection"
        )
        geo_converter = GeoProjectProcessor(
            geo_col="global_projection",
            from_format="EPSG:4326",
            to_format="EPSG:3857"
        )
        footprints = shapely_converter(dataset=footprints)
        footprints = geo_converter(dataset=footprints)

        #----------CREATE BUILDINGS----------
        buildings_creator = FootprintsToBuildingProcessor(
            polygons_col="global_projection",
            bottom_elevation_col="footprint_min_elevation",
            top_elevation_col="footprint_max_elevation",
            building_id_col=self.building_id_col,
            output_col="building"
        )
        buildings = buildings_creator(dataset=footprints)

        buildings = pd.DataFrame(buildings)
        buildings = buildings[["building"]]
        buildings["id"] = [str(i) for i in range(len(buildings))]

        def is_valid(row):
            for e in row["building"]:
                if e["top_elevation"] - e["bottom_elevation"] < 1e-8:
                    return False
            return True
        buildings = buildings[buildings.progress_apply(lambda row: is_valid(row), axis=1)]
        
        return buildings.to_dict("list")