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
    OrderIdCreator
)
from src.dataset_creator import (
    CsvDatasetCreator,
    PreprocessDatasetCreator
)
from src.dataset_creator import InversedJsonDatasetLoader

def main():
    footprint_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/171025_total_intersection_resaving/footprint_dataset.json"
    
    block_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/171025_total_intersection_resaving/block_dataset.json"
    property_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/171025_total_intersection_resaving/property_dataset.json"
    
    dwelling_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/residential-dwellings.csv"
    building_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/buildings-with-name-age-size-accessibility-and-bicycle-facilities.csv"
    address_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/street-addresses.csv"
    business_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/business-establishments-with-address-and-industry-classification.csv"
    bar_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/bars-and-pubs-with-patron-capacity.csv"
    cafe_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/data/melbourne_data/cafes-and-restaurants-with-seating-capacity.csv"

    output = {}
    tqdm.pandas()

    #Load datasets
    polygon_datasets = {}
    paths = [footprint_dataset_path, block_dataset_path, property_dataset_path]
    keys = ["footprint", "block", "property"]
    for i, k in enumerate(keys):
        path = paths[i]
        loader = InversedJsonDatasetLoader(path=path)
        dataset_dict = loader()
        dataset = pd.DataFrame(dataset_dict)
        dataset["id_buffer"] = dataset["id"]
        dataset = dataset.set_index("id_buffer")
        polygon_datasets[k] = dataset
    
    #Filter self-intersections
    for main_key, main_dataset in tqdm(polygon_datasets.items()):
        for intersect_key, intersect_dataset in polygon_datasets.items():
            if intersect_key == main_key:
                main_dataset[f"{intersect_key}_intersections"] = main_dataset.apply(lambda row: [spec for spec in row[f"{intersect_key}_intersections"] if spec["id"] != row["id"]], axis=1)
                polygon_datasets[main_key] = main_dataset
    
    #Analyze zero areas
    zero_areas = {}
    for main_key, main_dataset in tqdm(polygon_datasets.items()):
        local_areas = {}
        for intersect_key, intersect_dataset in polygon_datasets.items():
            main_dataset["zero_area"] = main_dataset.apply(lambda row: sum([spec["area"] for spec in row[f"{intersect_key}_intersections"]]) == 0, axis=1)
            local_areas[intersect_key] = main_dataset["zero_area"].value_counts().to_dict()
        zero_areas[main_key] = local_areas
    output["zero_areas"] = zero_areas

    #Analyse areas quantile
    def get_quantile(row):
        quantiles = [q/100 for q in range(100)]
        max_rel_area = max([spec["area"] for spec in row[f"{intersect_key}_intersections"]] + [0]) / row["area"]
        quantile = 0
        for q in quantiles:
            if max_rel_area >= q:
                quantile = q
        return quantile
    quantiles = {}
    for main_key, main_dataset in tqdm(polygon_datasets.items()):
        local_quantiles = {}
        for intersect_key, intersect_dataset in polygon_datasets.items():
            main_dataset["quantile"] = main_dataset.apply(lambda row: get_quantile(row), axis=1)
            local_quantiles[intersect_key] = main_dataset["quantile"].value_counts().to_dict()
        quantiles[main_key] = local_quantiles
    output["quantiles"] = quantiles

    #Analyse metadata distribution for different area thresholds
    #Load metadata datasets
    paths = [dwelling_dataset_path, building_dataset_path, business_dataset_path, cafe_dataset_path, bar_dataset_path]
    keys = ["dwelling", "building", "business", "cafe", "bar"]
    drop_cols = {
        "dwelling":["Block ID", "CLUE small area", "Longitude", "Latitude", "location"],
        "building":["Block ID", "Longitude", "Latitude", "location"],
        "business":["Block ID", "CLUE small area", "Industry (ANZSIC4) code", "Longitude", "Latitude", "Point"],
        "cafe":["Block ID", "CLUE small area", "Industry (ANZSIC4) code", "Longitude", "Latitude", "location"],
        "bar":["Block ID", "CLUE small area", "Longitude", "Latitude", "location"]
    }
    metadata_datasets = {}
    for i in tqdm(range(len(paths))):
        path = paths[i]
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
                    )
                ]
            )
        )
        dataset = dataset_loader()
        dataset = pd.DataFrame(dataset)
        local_drop = drop_cols[key]
        dataset = dataset.drop(columns=local_drop)
        #Get only newest records
        max_year = max(dataset["Census year"].values.tolist())
        dataset = dataset[dataset["Census year"] == max_year]
        metadata_datasets[key] = dataset
    
    thresholds = list(range(10))
    thresholds = [t/10 for t in thresholds]
    metadata_distribution = {}
    for t in tqdm(thresholds):
        local_data = {}
        #Match footprints with properties
        footprint_dataset = polygon_datasets["footprint"]
        footprint_dataset["properties"] = footprint_dataset.apply(lambda row: [spec["id"] for spec in row["property_intersections"] if spec["area"]/row["area"] > t], axis=1)
        def get_gt_properties(row):
            local_ids = row["properties"]
            total_ids = [polygon_datasets["property"]["Property_ID"].loc[i] for i in local_ids]
            total_ids = list(set(total_ids))
            return total_ids
        footprint_dataset["gt_properties"] = footprint_dataset.apply(lambda row: get_gt_properties(row), axis=1)

        #Group footprints to structures
        footprint_dataset["date_structure_id"] = footprint_dataset.apply(lambda row: f"{row['structure_id']}_{row['date_captured']}", axis=1)
        structure_dataset = footprint_dataset.groupby('date_structure_id').agg(list)
        flat_cols = ["structure_id", "date_captured"]
        for col in flat_cols:
            structure_dataset[col] = structure_dataset.apply(lambda row: row[col][0], axis=1)
        def sum_gt_properties(row):
            total_properties = []
            for plist in row["gt_properties"]:
                total_properties.extend(plist)
            total_properties = list(set(total_properties))
            return total_properties
        structure_dataset["gt_properties"] = structure_dataset.apply(lambda row: sum_gt_properties(row), axis=1)
        structure_dataset["n_gt_properties"] = structure_dataset.apply(lambda row: len(row["gt_properties"]), axis=1)
        local_data["n_gt_properties_per_structure"] = structure_dataset["n_gt_properties"].value_counts().to_dict()

        #Merge metadata to structures
        for key in keys:
            dataset = metadata_datasets[key]
            structure_dataset[key] = structure_dataset.apply(lambda row: dataset[dataset["Property ID"].isin(row["gt_properties"])], axis=1)
        structure_metadata = {}
        for key in keys:
            structure_dataset["n"] = structure_dataset.apply(lambda row: len(row[key]), axis=1)
            structure_metadata[key] = structure_dataset["n"].value_counts().to_dict()
        local_data["structure_metadata"] = structure_metadata
        metadata_distribution[t] = local_data
    output["metadata_disctibution"] = metadata_distribution
    

    with open("/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/171025_total_intersection_analysis/output.json", "w+") as f:
        json.dump(output, f)

if __name__ == "__main__":
    main()