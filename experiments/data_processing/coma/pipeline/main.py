import json
import math
import os
from copy import deepcopy

import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, Polygon, MultiPolygon
import shapely

import sys
sys.path.append("/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation")
from src.interpretable_function import InterpretableFunction
from src.core.function_utils import FunctionWrapper, FunctionGraph
from src.core.parsers import (
    DictToDictParser, AnyToDictParser, DictToAnyParser
)
from src.dataset_creator import (
    CsvDatasetCreator,
    PreprocessDatasetCreator
)
from src.dataset_processor import (
    GeoJSONToShapelyProcessor,
    GeoProjectProcessor,
    CompositionProcessor,
    FunctionProcessor,
    FootprintsToMassingsProcessor
)
from src.dataset_merger import GeoPandasDatasetIntersector
from src.dataset_handler import InversedJsonDatasetSaver

def main():
    footprint_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/2023-building-footprints.csv"
    
    block_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/blocks-for-census-of-land-use-and-employment-clue.csv"
    property_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/property-boundaries.csv"
    
    dwelling_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/residential-dwellings.csv"
    building_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/buildings-with-name-age-size-accessibility-and-bicycle-facilities.csv"
    address_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/street-addresses.csv"
    business_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/business-establishments-with-address-and-industry-classification.csv"
    bar_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/bars-and-pubs-with-patron-capacity.csv"
    cafe_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/cafes-and-restaurants-with-seating-capacity.csv"

    save_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/101025_footprints_to_all_geo_graph/footprint_to_all_geo_graph.json"

    #--------------------POLYGONS GEO MERGING--------------------
    print("Prepare polygon datasets")
    paths = [footprint_dataset_path, block_dataset_path, property_dataset_path]
    id_cols = ["OBJECTID", "block_id", "Gis_ID"]
    keys = ["footprint", "block", "property"]
    polygon_datasets = {}
    for i in range(len(paths)):
        path = paths[i]
        id_col = id_cols[i]
        key = keys[i]
        dataset_loader = PreprocessDatasetCreator(
            base_creator=CsvDatasetCreator(
                path=path,
                pandas_args={
                    "sep":",",
                    "index_col":False
                }
            ),
            preprocessor=CompositionProcessor(
                base_processors=[
                    FunctionProcessor(
                        function=FunctionWrapper(
                            function=InterpretableFunction(
                                function="str(id)",
                                returns_output=True
                            ),
                            input_parser=DictToDictParser(
                                keys_mapping={"id":"id"}
                            )
                        ),
                        cols_to_args_mapping={
                            id_col:"id"
                        },
                        output_key="id"
                    ),
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
        dataset = dataset_loader()
        dataset = pd.DataFrame(dataset)
        polygon_datasets[key] = dataset

    print("Intersect polygon datasets")
    for main_key in polygon_datasets.keys():
        for join_key in polygon_datasets.keys():
            intersector = GeoPandasDatasetIntersector(
                intersect_dataset_key="intersect_dataset"
                intersect_dataset_geo_col="global_projection",
                intersect_dataset_id_col="id",
                main_dataset_key="main_dataset",
                main_dataset_geo_col="global_projection",
                main_dataset_id_col="id",
                drop_self_intersections=True if main_key == join_key else False,
                output_properties=["id", "area"],
                output_col=f"{join_key}_intersections"
            )
            datasets = {
                "intersect_dataset": polygon_datasets[join_key].to_dict("list"),
                "main_dataset": polygon_datasets[main_key].to_dict("list")
            }
            new_main_dataset = intersector(datasets=datasets)
            polygon_datasets[main_key] = pd.DataFrame(new_main_dataset)
    
    #--------------------SITES CREATION--------------------
    
    #--------------------MASSINGS CREATION--------------------
    footprints_to_massings_processor = CompositionProcessor(
        base_processors=[
            FunctionProcessor(
                InterpretableFunction(
                    function="f'{id}_{date}'",
                    returns_output=True
                ),
                cols_to_args_mapping={
                    "id":"structure_id",
                    "date":"date_captured"
                },
                output_key="building_id"
            ),
            FootprintsToMassingsProcessor(
                polygons_col="global_projection",
                bottom_elevation_col="footprint_min_elevation",
                top_elevation_col="footprint_max_elevation",
                building_id_col="building_id",
                output_col="massing"
            )
        ]
    )
    """
    1. Group footprints
    2. Create massings
    3. Filter invalid geometry
    """

    #--------------------REQUIREMENTS CREATION--------------------

    #--------------------CONTEXT CREATION--------------------

if __name__ == "__main__":
    main()