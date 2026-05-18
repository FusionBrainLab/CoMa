import json
import math
import os
import random
from copy import copy

import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, Polygon, MultiPolygon
import trimesh
import shapely

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
from src.dataset_handler import InversedJsonDatasetSaver
from src.dataset_creator import InversedJsonDatasetLoader
from src.polygons_to_shapely_converter import PolygonsToShapelyConverter

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
    print("Load polygon datasets")
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
    for main_key, main_dataset in polygon_datasets.items():
        for intersect_key, intersect_dataset in polygon_datasets.items():
            if intersect_key == main_key:
                main_dataset[f"{intersect_key}_intersections"] = main_dataset.apply(lambda row: [spec for spec in row[f"{intersect_key}_intersections"] if spec["id"] != row["id"]], axis=1)
                polygon_datasets[main_key] = main_dataset

    print("Prepare properties")
    #Leave top-level properties
    property_dataset = polygon_datasets["property"]
    def is_parent_property(row):
        intersections = row["property_intersections"]
        area = row["area"]
        is_top = True
        for intersection in intersections:
            local_area = property_dataset.loc[intersection["id"]]["area"]
            intersection_area = intersection["area"]
            if intersection_area/area > intersection_area/local_area:
                is_top = False
                break
        return is_top
    final_dataset = property_dataset[property_dataset.progress_apply(lambda row: is_parent_property(row), axis=1)]

    #Get child properties
    final_dataset["child_properties"] = final_dataset.progress_apply(lambda row: [spec["id"] for spec in row["property_intersections"] if spec["area"] > 0], axis=1)

    #Get surrounding properties
    final_dataset["surrounding_properties"] = final_dataset.progress_apply(lambda row: [spec["id"] for spec in row["property_intersections"] if spec["area"] == 0], axis=1)

    #Get parent blocks
    def get_parent_block(row):
        block_id = None
        max_area = 0
        for spec in row["block_intersections"]:
            if spec["area"] > max_area:
                block_id = spec["id"]
                max_area = spec["area"]
        return block_id
    final_dataset["parent_block"] = final_dataset.progress_apply(lambda row: get_parent_block(row), axis=1)

    #Get surrounding blocks
    final_dataset["surrounding_blocks"] = final_dataset.progress_apply(lambda row: [spec["id"] for spec in polygon_datasets["block"]["block_intersections"].loc[row["parent_block"]] if spec["area"] == 0], axis=1)

    #Get surrounding blocks with expanded radius
    def expand_surrounding_blocks(row):
        cur_blocks = row["surrounding_blocks"]
        new_blocks = []
        for block in cur_blocks:
            new_blocks.append(block)
            local_blocks = [spec["id"] for spec in polygon_datasets["block"]["block_intersections"].loc[block] if spec["area"] == 0 and spec["id"] not in new_blocks and spec["id"] != row["parent_block"]]
            new_blocks.extend(local_blocks)
        return new_blocks
    final_dataset["surrounding_blocks"] = final_dataset.progress_apply(lambda row: expand_surrounding_blocks(row), axis=1)

    print("Create massings")
    #Match footprints with properties
    footprint_dataset = polygon_datasets["footprint"]
    geo_match_threshold = 0.9
    footprint_dataset["properties"] = footprint_dataset.progress_apply(lambda row: [spec["id"] for spec in row["property_intersections"] if spec["area"]/row["area"] > geo_match_threshold], axis=1)
    
    #Get property ids for footprints to match with metadata
    def get_gt_properties(row):
        local_ids = row["properties"]
        total_ids = [polygon_datasets["property"]["Property_ID"].loc[i] for i in local_ids]
        total_ids = list(set(total_ids))
        return total_ids
    footprint_dataset["gt_properties"] = footprint_dataset.progress_apply(lambda row: get_gt_properties(row), axis=1)

    #Filter footprints with zero height
    footprint_dataset = footprint_dataset[footprint_dataset.apply(lambda row: row["footprint_min_elevation"] != row["footprint_max_elevation"], axis=1)]

    #Group footprints to structures
    footprint_dataset["date_structure_id"] = footprint_dataset.progress_apply(lambda row: f"{row['structure_id']}_{row['date_captured']}", axis=1)
    structure_dataset = footprint_dataset.groupby('date_structure_id').agg(list)
    
    structure_dataset["footprint_ids"] = structure_dataset["id"]
    structure_dataset["footprint_types"] = structure_dataset["footprint_type"]

    #Flatten some cols
    flat_cols = ["structure_id", "date_captured"]
    for col in flat_cols:
        structure_dataset[col] = structure_dataset.apply(lambda row: row[col][0], axis=1)
    def sum_properties(properties):
        total_properties = []
        for plist in properties:
            total_properties.extend(plist)
        total_properties = list(set(total_properties))
        return total_properties
    structure_dataset["properties"] = structure_dataset.progress_apply(lambda row: sum_properties(row["properties"]), axis=1)
    structure_dataset["gt_properties"] = structure_dataset.progress_apply(lambda row: sum_properties(row["gt_properties"]), axis=1)

    #Create massings
    def get_massing(row):
        bottom_elevations = row["footprint_min_elevation"]
        top_elevations = row["footprint_max_elevation"]
        min_elevation = min(bottom_elevations)
        bottom_elevations = [e - min_elevation for e in bottom_elevations]
        top_elevations = [e - min_elevation for e in top_elevations]
        polygons = row["global_projection"]
        massing = []
        for plist, bottom, top in zip(polygons, bottom_elevations, top_elevations):
            extrusion = {
                "polygons":[[tuple(p) for p in polygon] for polygon in plist],
                "bottom_elevation":bottom,
                "top_elevation":top
            }
            massing.append(extrusion)
        return massing
    structure_dataset["massing"] = structure_dataset.progress_apply(lambda row: get_massing(row), axis=1)

    """#Create meshes
    print("Create meshes")
    mesh_creator = MassingToTrimeshConverter()
    structure_dataset["mesh"] = structure_dataset.progress_apply(lambda row: mesh_creator(massing=row["massing"]), axis=1)"""

    print("Load metadata datasets")
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
    
    #Merge metadata to structures
    for key in tqdm(keys):
        dataset = metadata_datasets[key]
        structure_dataset[key] = structure_dataset.apply(lambda row: dataset[dataset["Property ID"].isin(row["gt_properties"])], axis=1)

    #Add metadata columns
    #Add columns and rename them for buildings and dwellings
    building_base_cols = [
        ["Dwelling type", "Dwelling number"],
        ["Building name", "Construction year", "Number of floors (above ground)", "Predominant space use", "Accessibility type description", "Accessibility rating"]
    ]
    building_new_cols = [
        ["building_function", "n_dwellings"],
        ["building_name", "construction_year", "n_floors", "building_function", "accessibility_type", "accessibility_rating"]
    ]
    building_keys = ["dwelling", "building"]
    for key, base_cols, new_cols in tqdm(zip(building_keys, building_base_cols, building_new_cols), total=len(building_keys)):
        structure_dataset[key] = structure_dataset.apply(lambda row: row[key][row[key].apply(lambda inrow: any([inrow[k] for k in base_cols]), axis=1)], axis=1)
        for base_col, new_col in zip(base_cols, new_cols):
            structure_dataset[new_col] = structure_dataset.apply(lambda row: row[key][base_col].iloc[0] if len(row[key]) > 0 else None, axis=1)
    
    #Count total commercial spaces
    structure_dataset["n_commercial_spaces"] = structure_dataset.progress_apply(lambda row: len(row["business"]), axis=1)
    #Filter only offices - spaces that are not represented in cafes and bars
    nonbusiness_spaces = set(metadata_datasets["cafe"]["Trading name"].values.tolist() + metadata_datasets["bar"]["Trading name"].values.tolist())
    structure_dataset["offices"] = structure_dataset.progress_apply(lambda row: row["business"][~row["business"].isin(nonbusiness_spaces)], axis=1)
    #Fill nan values in offices
    office_replacement = {
        "Trading name":"",
        "Industry (ANZSIC4) description":""
    }
    def office_fillna(row):
        offices = row["offices"]
        for k, v in office_replacement.items():
            offices[k] = offices[k].fillna(v)
        return offices
    structure_dataset["offices"] = structure_dataset.progress_apply(lambda row: office_fillna(row), axis=1)
    #Rename offices columns
    structure_dataset["offices"] = structure_dataset.progress_apply(lambda row: row["offices"][["Trading name", "Industry (ANZSIC4) description"]].rename(columns={"Trading name":"name", "Industry (ANZSIC4) description":"function"}).to_dict(orient='records'), axis=1)

    #Get non-office public spaces - spaces with known public capacity
    def get_public_spaces(row):
        base_spaces = row["cafe"]
        if len(base_spaces) == 0:
            return []
        base_spaces["name"] = base_spaces["Trading name"]
        base_spaces = base_spaces.groupby('name').agg(list)
        base_spaces["name"] = base_spaces.apply(lambda inrow: inrow["Trading name"][0], axis=1).fillna("")
        base_spaces["function"] = base_spaces.apply(lambda inrow: inrow["Industry (ANZSIC4) description"][0], axis=1).fillna("")
        base_spaces["indoor_capacity"] = base_spaces.apply(lambda inrow: inrow["Number of seats"][inrow["Seating type"].index("Seats - Indoor")] if "Seats - Indoor" in inrow["Seating type"] else 0, axis=1).fillna(0)
        base_spaces["outdoor_capacity"] = base_spaces.apply(lambda inrow: inrow["Number of seats"][inrow["Seating type"].index("Seats - Outdoor")] if "Seats - Outdoor" in inrow["Seating type"] else 0, axis=1).fillna(0)
        base_spaces = base_spaces[["name", "function", "indoor_capacity", "outdoor_capacity"]]
        spaces = base_spaces.to_dict(orient='records')
        return spaces
    structure_dataset["public_spaces"] = structure_dataset.progress_apply(lambda row: get_public_spaces(row), axis=1)
    structure_dataset = structure_dataset.drop(columns=[
        "footprint_type", "footprint_max_elevation", "footprint_min_elevation", "id", "geo_projection", "global_projection", "area",
        "footprint_intersections", "block_intersections", "property_intersections", "gt_properties", "dwelling",
        "building", "business", "cafe", "bar", "footprint_ids", "footprint_types"
    ])

    #Replace nan values
    replacement = {
        "building_function":"",
        "n_dwellings":0,
        "building_name":"",
        "construction_year":-1,
        "n_floors":0,
        "accessibility_type":"",
        "accessibility_rating":-1
    }
    for k, v in replacement.items():
        structure_dataset[k] = structure_dataset[k].fillna(v)

    #Compute floor height
    def get_floor_height(row):
        n_floors = row["n_floors"]
        massing = row["massing"]
        height = max([e["top_elevation"] for e in massing])
        floor_height = round(height/n_floors, 3) if n_floors > 0 else height
        return floor_height
    structure_dataset["floor_height"] = structure_dataset.progress_apply(lambda row: get_floor_height(row), axis=1)

    polygons_converter = PolygonsToShapelyConverter()
    #Compute area
    def get_usable_area(row):
        n_floors = int(row["n_floors"])
        floor_height = row["floor_height"]
        massing = row["massing"]
        area = 0
        heights = [0] + [floor_height * (i + 1) for i in range(n_floors - 1)] if n_floors > 1 else [0]
        for local_height in heights:
            local_footprints = [e["polygons"] for e in massing if local_height >= e["bottom_elevation"] and local_height < e["top_elevation"]]
            local_footprints = [polygons_converter(polygons=p) for p in local_footprints]
            if len(local_footprints) == 0:
                continue
            local_floor = local_footprints[0]
            for i in range(1, len(local_footprints)):
                local_floor = local_floor.union(local_footprints[i])
            local_area = shapely.area(local_floor)
            area += local_area
        return area    
    structure_dataset["usable_area"] = structure_dataset.progress_apply(lambda row: get_usable_area(row), axis=1)

    #Create dataset
    print("Create dataset")

    """#Normalize site contour
    final_dataset["base_point"] = final_dataset.progress_apply(lambda row: row["global_projection"][0][0], axis=1)
    def normalize_site(row):
        cur_site = row["global_projection"]
        base_point = row["base_point"]
        new_site = [[(p[0] - base_point[0], p[1] - base_point[1]) for p in polygon] for polygon in cur_site]
        return new_site
    final_dataset["site_contour"] = final_dataset.progress_apply(lambda row: normalize_site(row), axis=1)"""
    
    final_dataset["geo_site_contour"] = final_dataset["geo_projection"]

    #Merge buildings into properties
    contains_value = np.vectorize(lambda x, val: val in x)
    final_dataset["buildings"] = final_dataset.progress_apply(lambda row: structure_dataset[contains_value(structure_dataset["properties"], row["id"])], axis=1)
    final_dataset = final_dataset[final_dataset.apply(lambda row: len(row["buildings"]) > 0, axis=1)]

    #Expand properties by building year consistency
    new_properties = {k: [] for k in final_dataset.columns.to_list()}
    def expand_buildings(row):
        local_buildings = pd.DataFrame(row["buildings"])
        local_ids = local_buildings["structure_id"].values.tolist()
        if len(set(local_ids)) == len(local_ids):
            return row["buildings"]
        duplicated_ids = copy(local_ids)
        for i in list(set(local_ids)):
            duplicated_ids.remove(i)
        duplicated_ids = list(set(duplicated_ids))

        base_dated_ids = local_buildings[~local_buildings["structure_id"].isin(duplicated_ids)].apply(lambda inrow: f"{inrow["structure_id"]}_{inrow["date_captured"]}", axis=1).values.tolist()

        duplicated_dates = [local_buildings[local_buildings["structure_id"] == i]["date_captured"].values.tolist() for i in duplicated_ids]
        duplicated_lists = [[f"{i}_{d}" for d in dates] for i, dates in zip(duplicated_ids, duplicated_dates)]
        duplicated_combinations = [duplicated_lists[0]]
        for i in range(1, len(duplicated_lists)):
            new_combinations = []
            for c in duplicated_combinations:
                for s in duplicated_lists[i]:
                    new_c = copy(c)
                    new_c.append(s)
                    new_combinations.append(new_c)
            duplicated_combinations = new_combinations
        
        total_building_combinations = [base_dated_ids + d for d in duplicated_combinations]
        cur_buildings = local_buildings[local_buildings.apply(lambda inrow: f"{inrow["structure_id"]}_{inrow["date_captured"]}" in total_building_combinations[0], axis=1)]

        for i in range(1, len(total_building_combinations)):
            for k in row.to_dict().keys():
                value = None
                if k == "buildings":
                    value = local_buildings[local_buildings.apply(lambda inrow: f"{inrow["structure_id"]}_{inrow["date_captured"]}" in total_building_combinations[i], axis=1)]
                else:
                    value = row[k]
                new_properties[k].append(value)

        return cur_buildings

    final_dataset["buildings"] = final_dataset.progress_apply(lambda row: expand_buildings(row), axis=1)
    new_properties = pd.DataFrame(new_properties)
    final_dataset = pd.concat([final_dataset, new_properties])

    #Parse metadata for properties buildings
    def get_requirements(row):
        buildings = pd.DataFrame(row["buildings"])
        buildings["id"] = list(range(len(buildings)))
        buildings = buildings.drop(columns=["massing", "properties", "date_captured", "structure_id"])
        requirements = buildings.to_dict(orient='records')
        return requirements
    final_dataset["requirements"] = final_dataset.progress_apply(lambda row: get_requirements(row), axis=1)
    
    #Collect massings for properties
    def get_massing(row):
        buildings = pd.DataFrame(row["buildings"])
        buildings["id"] = list(range(len(buildings)))
        buildings = buildings[["id", "massing"]]
        """base_point = row["base_point"]
        buildings["massing"] = buildings.apply(lambda inrow: [{"polygons":[[(p[0] - base_point[0], p[1] - base_point[1]) for p in polygon] for polygon in inrow["massing"][i]["polygons"]], "bottom_elevation":inrow["massing"][i]["bottom_elevation"], "top_elevation":inrow["massing"][i]["top_elevation"]} for i in range(len(inrow["massing"]))], axis=1)"""
        massing = buildings.to_dict(orient='records')
        return massing
    final_dataset["massing"] = final_dataset.progress_apply(lambda row: get_massing(row), axis=1)

    """def get_mesh(row):
        buildings = row["buildings"]
        mesh = trimesh.util.concatenate(buildings["mesh"].values.tolist())
        return mesh
    final_dataset["mesh"] = final_dataset.progress_apply(lambda row: get_mesh(row), axis=1)"""
    
    #Collect surrounding properties for form the environment
    block_to_properties = {}
    for block in tqdm(final_dataset['parent_block'].unique()):
        block_properties = final_dataset[final_dataset['parent_block'] == block]
        block_to_properties[block] = block_properties
    def get_env_properties(row):
        surrounding_blocks = row["surrounding_blocks"]
        current_id = row["id"]
        parent_block = row["parent_block"]
        
        properties = []
        for block in surrounding_blocks:
            if block in block_to_properties:
                properties.extend(block_to_properties[block]["id"].values.tolist())
        
        if parent_block in block_to_properties:
            parent_properties = block_to_properties[parent_block][block_to_properties[parent_block]["id"] != current_id]["id"].values.tolist()
            properties.extend(parent_properties)
        
        properties = list(set(properties))
        return properties
    final_dataset["env_properties"] = final_dataset.progress_apply(lambda row: get_env_properties(row), axis=1)

    final_dataset["name"] = final_dataset["Property_Name"].fillna("")
    final_dataset["global_site_contour"] = final_dataset["global_projection"]
    final_dataset = final_dataset.drop(columns=[
        "address", "Property_ID", "Property_Name", "Date_Updated", "footprint_intersections", "block_intersections", "property_intersections",
        "buildings", "geo_projection", "global_projection", "area", "child_properties", "parent_block",
        "surrounding_blocks"
    ])

    #Expand properties by building year consistency
    

    #Save dataset
    print("Save dataset")
    """folder_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/171025_data_pipeline/dataset/surrounding_mesh"
    final_dataset.progress_apply(lambda row: row["surrounding_mesh"].export(os.path.join(folder_path, f"{row["id"]}.obj")), axis=1)
    final_dataset["surrounding_mesh"] = final_dataset.apply(lambda row: os.path.join(folder_path, f"{row["id"]}.obj"), axis=1)
    folder_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/171025_data_pipeline/dataset/mesh"
    final_dataset.progress_apply(lambda row: row["mesh"].export(os.path.join(folder_path, f"{row["id"]}.obj")), axis=1)
    final_dataset["mesh"] = final_dataset.apply(lambda row: os.path.join(folder_path, f"{row["id"]}.obj"), axis=1)"""

    dataset_dict = final_dataset.to_dict("list")
    saver = InversedJsonDatasetSaver(path="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/171025_data_pipeline/dataset.json")
    saver(dataset=dataset_dict)
    
if __name__ == "__main__":
    main()