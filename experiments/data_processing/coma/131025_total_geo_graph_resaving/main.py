import json
import math
import os

import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, Polygon, MultiPolygon

import sys
sys.path.append("/workspace-SR008.fs2/maslov/massing_generation")
from src.dataset_processor import (
    CompositionProcessor,
    FunctionProcessor,
    DatasetFilter,
    OrderIdCreator
)
from src.dataset_creator import (
    CsvDatasetCreator,
    PreprocessDatasetCreator
)
from src.interpretable_function import InterpretableFunction
from src.point_geo_projector import PointGeoProjector
from src.core.function_utils import FunctionWrapper, FunctionGraph
from src.core.parsers import (
    DictToDictParser, AnyToDictParser, DictToAnyParser
)
from src.polys_geo_projector import PolysGeoProjector
from src.md_point_converter import MDPointConverter
from src.md_polygons_converter import MDPolygonsConverter
from src.dataset_handler import InversedJsonDatasetSaver
from src.dataset_creator import InversedJsonDatasetLoader
from src.polygons_to_shapely_converter import PolygonsToShapelyConverter
from src.point_to_shapely_converter import PointToShapelyConverter
from src.geo_pandas_id_dataset_joiner import GeoPandasIdDatasetJoiner

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
    point_cols = ["Geo Point", "Geo Point", "geo_point_2d"]
    keys = ["footprint", "block", "property"]
    polygon_datasets = {}
    for i in range(len(paths)):
        path = paths[i]
        id_col = id_cols[i]
        point_col = point_cols[i]
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
            ),
            FunctionProcessor(
                function=FunctionWrapper(
                    function=MDPointConverter(),
                    input_parser=DictToDictParser(
                        keys_mapping={"md_point":"md_point"}
                    )
                ),
                cols_to_args_mapping={
                    point_col:"md_point"
                },
                dataset_arg="dataset",
                output_key="geo_point"
            ),
            FunctionProcessor(
                function=FunctionWrapper(
                    function=PointGeoProjector(),
                    input_parser=DictToDictParser(
                        keys_mapping={"point":"geo_point"}
                    )
                ),
                cols_to_args_mapping={
                    "geo_point":"geo_point"
                },
                dataset_arg="dataset",
                output_key="global_point"
            ),
            FunctionProcessor(
                function=FunctionWrapper(
                    function=PointToShapelyConverter(),
                    input_parser=DictToDictParser(
                        keys_mapping={"point":"global_point"}
                    )
                ),
                cols_to_args_mapping={
                    "global_point":"global_point"
                },
                dataset_arg="dataset",
                output_key="shapely_global_point"
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
        polygon_datasets[key] = dataset

    #Join polygon datasets
    print("Join polygon datasets")
    predicates = ["contains", "within", "touches", "overlaps", "covers", "covered_by", "crosses", "intersects"]
    for main_key in tqdm(polygon_datasets.keys(), total=len(list(polygon_datasets.keys()))):
        for join_key in polygon_datasets.keys():
            for p in predicates:
                joiner = GeoPandasIdDatasetJoiner(
                    join_dataset_geo_col="shapely_global_projection",
                    join_dataset_id_col="id",
                    main_dataset_geo_col="shapely_global_projection",
                    predicate=p,
                    output_col=f"{join_key}_{p}_ids"
                )
                new_main_dataset = joiner(main_dataset=polygon_datasets[main_key], join_dataset=polygon_datasets[join_key])
                polygon_datasets[main_key] = new_main_dataset
            joiner = GeoPandasIdDatasetJoiner(
                join_dataset_geo_col="shapely_global_point",
                join_dataset_id_col="id",
                main_dataset_geo_col="shapely_global_projection",
                predicate="intersects",
                output_col=f"{join_key}_point_intersects_ids"
            )
            new_main_dataset = joiner(main_dataset=polygon_datasets[main_key], join_dataset=polygon_datasets[join_key])
            polygon_datasets[main_key] = new_main_dataset
    
    #Save polygon datasets
    print("Save polygon datasets")
    drop_cols = {
        "footprint":["shapely_global_projection", "shapely_global_point", "Geo Point", "Geo Shape", "OBJECTID", "roof_type", "structure_max_elevation", "structure_min_elevation", "footprint_extrusion", "structure_extrusion"],
        "block":["shapely_global_projection", "shapely_global_point", "Geo Shape", "Geo Point", "block_id", "clue_area"],
        "property":["shapely_global_projection", "shapely_global_point", "Geo Shape", "poly_area", "Gis_ID", "Polygon", "geo_point_2d"]
    }
    folder_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/131025_total_geo_graph_resaving"
    for key, dataset in tqdm(polygon_datasets.items(), total=len(list(polygon_datasets.keys()))):
        local_drop = drop_cols[key]
        pd_dataset = pd.DataFrame(dataset)
        pd_dataset = pd_dataset.drop(columns=local_drop)
        dataset = pd_dataset.to_dict("list")
        path = os.path.join(folder_path, f"{key}_dataset.json")
        saver = InversedJsonDatasetSaver(path=path)
        saver(dataset=dataset)

    #Prepare point datasets
    print("Prepare point datasets")
    paths = [dwelling_dataset_path, building_dataset_path, address_dataset_path, business_dataset_path, cafe_dataset_path, bar_dataset_path]
    keys = ["dwelling", "building", "address", "business", "cafe", "bar"]
    point_cols = ["location", "location", "Geo Point", "Point", "location", "location"]
    point_datasets = {}
    for i in range(len(paths)):
        path = paths[i]
        point_col = point_cols[i]
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
                    OrderIdCreator(
                        field_name="id"
                    ),
                    DatasetFilter(
                        filter=InterpretableFunction(
                            function="type(location) == str",
                            returns_output=True
                        ),
                        cols_to_args_mapping={
                            point_col:"location"
                        }
                    ),
                    FunctionProcessor(
                        function=FunctionWrapper(
                            function=MDPointConverter(),
                            input_parser=DictToDictParser(
                                keys_mapping={"md_point":"md_point"}
                            )
                        ),
                        cols_to_args_mapping={
                            point_col:"md_point"
                        },
                        dataset_arg="dataset",
                        output_key="geo_point"
                    ),
                    FunctionProcessor(
                        function=FunctionWrapper(
                            function=PointGeoProjector(),
                            input_parser=DictToDictParser(
                                keys_mapping={"point":"geo_point"}
                            )
                        ),
                        cols_to_args_mapping={
                            "geo_point":"geo_point"
                        },
                        dataset_arg="dataset",
                        output_key="global_point"
                    ),
                    FunctionProcessor(
                        function=FunctionWrapper(
                            function=PointToShapelyConverter(),
                            input_parser=DictToDictParser(
                                keys_mapping={"point":"global_point"}
                            )
                        ),
                        cols_to_args_mapping={
                            "global_point":"global_point"
                        },
                        dataset_arg="dataset",
                        output_key="shapely_global_point"
                    )
                ]
            )
        )
        dataset = dataset_loader()
        point_datasets[key] = dataset
    
    #Join point datasets
    print("Join point datasets")
    for point_key in tqdm(point_datasets.keys(), total=len(list(point_datasets.keys()))):
        for polygon_key, polygon_dataset in polygon_datasets.items():
            joiner = GeoPandasIdDatasetJoiner(
                join_dataset_geo_col="shapely_global_projection",
                join_dataset_id_col="id",
                main_dataset_geo_col="shapely_global_point",
                predicate="intersects",
                output_col=f"{polygon_key}_ids"
            )
            new_point_dataset = joiner(main_dataset=point_datasets[point_key], join_dataset=polygon_dataset)
            point_datasets[point_key] = new_point_dataset
    
    #Save point datasets
    print("Save point datasets")
    drop_cols = {
        "dwelling":["shapely_global_point", "Block ID", "CLUE small area", "Longitude", "Latitude", "location"],
        "building":["shapely_global_point", "Block ID", "Longitude", "Latitude", "location"],
        "address":["shapely_global_point", "Geo Point", "Geo Shape", "suburb_id", "latitude", "easting", "northing", "gisid", "longitude", "suburb", "street_id", "add_comp"],
        "business":["shapely_global_point", "Block ID", "CLUE small area", "Industry (ANZSIC4) code", "Longitude", "Latitude", "Point"],
        "cafe":["shapely_global_point", "Block ID", "CLUE small area", "Industry (ANZSIC4) code", "Longitude", "Latitude", "location"],
        "bar":["shapely_global_point", "Block ID", "CLUE small area", "Longitude", "Latitude", "location"]
    }
    folder_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/131025_total_geo_graph_resaving"
    for key, dataset in tqdm(point_datasets.items(), total=len(list(point_datasets.keys()))):
        local_drop = drop_cols[key]
        pd_dataset = pd.DataFrame(dataset)
        pd_dataset = pd_dataset.drop(columns=local_drop)
        dataset = pd_dataset.to_dict("list")
        path = os.path.join(folder_path, f"{key}_dataset.json")
        saver = InversedJsonDatasetSaver(path=path)
        saver(dataset=dataset)

if __name__ == "__main__":
    main()