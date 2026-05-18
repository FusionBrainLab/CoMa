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
sys.path.append("/workspace-SR008.fs2/maslov/massing_generation")
from src.dataset_processor import (
    CompositionProcessor,
    FunctionProcessor,
)
from src.dataset_creator import (
    CsvDatasetCreator,
    PreprocessDatasetCreator
)
from src.interpretable_function import InterpretableFunction
from src.core.function_utils import FunctionWrapper, FunctionGraph
from src.core.parsers import (
    DictToDictParser, AnyToDictParser, DictToAnyParser
)
from src.polys_geo_projector import PolysGeoProjector
from src.md_polygons_converter import MDPolygonsConverter
from src.dataset_handler import InversedJsonDatasetSaver
from src.polygons_to_shapely_converter import PolygonsToShapelyConverter
from src.geo_pandas_dataset_intersector import GeoPandasDatasetIntersector

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

    #Prepare polygon datasets
    print("Prepare polygon datasets")
    paths = [footprint_dataset_path, block_dataset_path, property_dataset_path]
    id_cols = ["OBJECTID", "block_id", "Gis_ID"]
    keys = ["footprint", "block", "property"]
    polygon_datasets = {}
    for i in range(len(paths)):
        path = paths[i]
        id_col = id_cols[i]
        key = keys[i]
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
                dataset_arg="dataset",
                output_key="id"
            ),
            FunctionProcessor(
                function=FunctionWrapper(
                    function=MDPolygonsConverter(),
                    input_parser=DictToDictParser(
                        keys_mapping={"md_geo_shape":"md_geo_shape"}
                    )
                ),
                cols_to_args_mapping={
                    "Geo Shape":"md_geo_shape"
                },
                dataset_arg="dataset",
                output_key="geo_projection"
            ),
            FunctionProcessor(
                function=FunctionWrapper(
                    function=PolysGeoProjector(),
                    input_parser=DictToDictParser(
                        keys_mapping={"polys":"geo_projection"}
                    )
                ),
                cols_to_args_mapping={
                    "geo_projection":"geo_projection"
                },
                dataset_arg="dataset",
                output_key="global_projection"
            ),
            FunctionProcessor(
                function=FunctionWrapper(
                    function=PolygonsToShapelyConverter(),
                    input_parser=DictToDictParser(
                        keys_mapping={"polygons":"global_projection"}
                    )
                ),
                cols_to_args_mapping={
                    "global_projection":"global_projection"
                },
                dataset_arg="dataset",
                output_key="shapely_global_projection"
            )
        ]
        dataset_loader = PreprocessDatasetCreator(
            base_creator=CsvDatasetCreator(
                path=path,
                pandas_args={
                    "sep":",",
                    "index_col":False
                }
            ),
            preprocessor=CompositionProcessor(
                base_processors=base_processors
            )
        )
        dataset = dataset_loader()
        dataset = pd.DataFrame(dataset)
        polygon_datasets[key] = dataset

    #Compute areas
    print("Compute areas")
    for key, dataset in polygon_datasets.items():
        dataset["area"] = dataset.apply(lambda row: shapely.area(row["shapely_global_projection"]), axis=1)

    #Intersect polygon datasets
    print("Intersect polygon datasets")
    for main_key in polygon_datasets.keys():
        for join_key in polygon_datasets.keys():
            intersector = GeoPandasDatasetIntersector(
                intersect_dataset_geo_col="shapely_global_projection",
                intersect_dataset_id_col="id",
                main_dataset_geo_col="shapely_global_projection",
                main_dataset_id_col="id",
                output_col=f"{join_key}_intersections"
            )
            new_main_dataset = intersector(main_dataset=polygon_datasets[main_key].to_dict("list"), intersect_dataset=polygon_datasets[join_key].to_dict("list"))
            polygon_datasets[main_key] = pd.DataFrame(new_main_dataset)
    
    #Compute intersection areas
    print("Compute intersection areas")
    def compute_areas(intersections):
        areas = []
        for intersection in intersections:
            area = shapely.area(intersection["geometry"])
            area = area if area else 0
            areas.append({"id":intersection["intersect_id"], "area":area})
        return areas
    for main_key in tqdm(polygon_datasets.keys(), total=len(list(polygon_datasets.keys()))):
        for join_key in polygon_datasets.keys():
            new_main_dataset = polygon_datasets[main_key]
            new_main_dataset[f"{join_key}_intersections"] = new_main_dataset.progress_apply(lambda row: compute_areas(row[f"{join_key}_intersections"]), axis=1)
            polygon_datasets[main_key] = new_main_dataset
    
    #Save polygon datasets
    print("Save polygon datasets")
    drop_cols = {
        "footprint":["shapely_global_projection", "Geo Point", "Geo Shape", "OBJECTID", "roof_type", "structure_max_elevation", "structure_min_elevation", "footprint_extrusion", "structure_extrusion"],
        "block":["shapely_global_projection", "Geo Shape", "Geo Point", "block_id", "clue_area"],
        "property":["shapely_global_projection", "Geo Shape", "poly_area", "Gis_ID", "Polygon", "geo_point_2d"]
    }
    folder_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/171025_total_intersection_resaving"
    for key, dataset in tqdm(polygon_datasets.items(), total=len(list(polygon_datasets.keys()))):
        local_drop = drop_cols[key]
        pd_dataset = dataset.drop(columns=local_drop)
        dataset = pd_dataset.to_dict("list")
        path = os.path.join(folder_path, f"{key}_dataset.json")
        saver = InversedJsonDatasetSaver(path=path)
        saver(dataset=dataset)

if __name__ == "__main__":
    main()