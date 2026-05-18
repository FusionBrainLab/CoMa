import json
import math
import os
import random
import io
from collections import deque
from copy import copy

import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, Polygon, MultiPolygon
import trimesh
import pyvista as pv
from PIL import Image

import sys
sys.path.append("/workspace-SR008.fs2/maslov/massing_generation")
from src.dataset_handler import InversedJsonDatasetSaver
from src.dataset_creator import InversedJsonDatasetLoader
from src.polygons_to_shapely_converter import PolygonsToShapelyConverter
from src.polygons_map_trace_creator import PolygonsMapTraceCreator
from src.shapely_to_polygons_converter import ShapelyToPolygonsConverter

def main():
    loader = InversedJsonDatasetLoader(path="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/171025_data_pipeline/dataset.json")
    dataset = loader()
    dataset = pd.DataFrame(dataset)
    dataset["id_buffer"] = dataset["id"]
    dataset = dataset.set_index("id_buffer")
    polygons_converter = PolygonsToShapelyConverter()
    shapely_converter = ShapelyToPolygonsConverter()
    map_visualizer = PolygonsMapTraceCreator(fill_color="red", line_color="black")

    base_dataset = dataset.copy()

    print("Filter dataset")
    dataset = dataset[dataset.apply(lambda row: len(row["massing"]) <= 5, axis=1)]
    dataset = dataset[dataset.apply(lambda row: "House/Townhouse" not in [b["building_function"] for b in row["requirements"]], axis=1)]
    #dataset = dataset[dataset.apply(lambda row: any([b["n_floors"] >= 3 for b in row["requirements"]]), axis=1)]

    print("Prepare property combinations")
    #Group properties
    n_properties = 4
    graph = {k: [p for p in dataset["surrounding_properties"].loc[k] if p in dataset["id"].values.tolist()] for k in dataset["id"].values.tolist()}

    def find_connected_combinations_bfs(graph, n):
        """
        More efficient BFS-based approach to find connected n-node combinations.
        """
        def get_connected_combinations_from_start(start_node, n):
            """Find all n-node connected combinations starting from a specific node."""
            result = []
            queue = deque()
            
            # Start with combinations containing only the start node
            queue.append(({start_node}, [start_node]))  # (current_set, path for ordering)
            
            while queue:
                current_set, path = queue.popleft()
                
                if len(current_set) == n:
                    result.append(current_set.copy())
                    continue
                
                # Get all neighbors of the current combination
                neighbors = set()
                for node in current_set:
                    if node not in graph:
                        continue
                    for neighbor in graph[node]:
                        if neighbor not in current_set:
                            neighbors.add(neighbor)
                
                # Add neighbors in a way that maintains connectivity and avoids duplicates
                for neighbor in neighbors:
                    if neighbor > start_node or len(path) == 0:  # Avoid duplicates
                        new_set = current_set | {neighbor}
                        new_path = path + [neighbor]
                        queue.append((new_set, new_path))
            
            return result
        
        result = []
        nodes = sorted(graph.keys())  # Sort to ensure consistent ordering
        
        for start_node in nodes:
            combinations = get_connected_combinations_from_start(start_node, n)
            # Convert to frozenset and use set to remove duplicates
            for comb in combinations:
                result.append(frozenset(comb))
        
        # Remove duplicates and return
        return [set(comb) for comb in set(result)]

    total_combinations = []
    for i in range(2, n_properties + 1):
        combinations = find_connected_combinations_bfs(graph, i)
        total_combinations.extend(combinations)

    #Reset id
    dataset["property_id"] = dataset["id"]
    dataset["id"] = list(range(len(dataset)))
    dataset["id_buffer"] = dataset["id"]
    dataset = dataset.set_index("id_buffer")

    property_to_ids = {}
    for i in list(set(dataset["property_id"].values.tolist())):
        ids = dataset[dataset["property_id"] == i]["id"].values.tolist()
        property_to_ids[i] = ids

    print("Create combined dataset")
    new_dataset = {k: [] for k in dataset.columns.to_list()}
    for c in tqdm(total_combinations):
        local_ids = [dataset[dataset["property_id"] == i]["id"].values.tolist() for i in c]
        local_combinations = [[]]
        for i in range(len(local_ids)):
            new_combinations = []
            for local_c in local_combinations:
                for s in local_ids[i]:
                    new_c = copy(local_c)
                    new_c.append(s)
                    new_combinations.append(new_c)
            local_combinations = new_combinations
        for local_c in local_combinations:
            subset = dataset.loc[list(local_c)]
            subset["i"] = [0] * len(subset)
            sample = subset.groupby('i').agg(list).iloc[0].to_dict()
            sums = ["requirements", "massing", "env_properties"]
            for s in sums:
                sample[s] = [e for l in sample[s] for e in l]
            sample["requirements"] = [{"id":str(i), **{k: v for k, v in sample["requirements"][i].items() if k != "id"}} for i in range(len(sample["requirements"]))]
            sample["massing"] = [{"id":str(i), **{k: v for k, v in sample["massing"][i].items() if k != "id"}} for i in range(len(sample["massing"]))]
            sample["env_properties"] = [p for p in set(sample["env_properties"]) if p not in sample["property_id"]]
            polys = ["geo_site_contour", "global_site_contour"]
            for p in polys:
                sample[p] = [[[(point[0], point[1]) for point in polygon] for polygon in plist] for plist in sample[p]]
                sample[p] = [polygons_converter(polygons=poly) for poly in sample[p]]
                total_p = sample[p][0]
                for i in range(1, len(sample[p])):
                    total_p = total_p.union(sample[p][i])
                sample[p] = shapely_converter(polygons=total_p)
            for k, v in sample.items():
                new_dataset[k].append(v)
    
    new_dataset = pd.DataFrame(new_dataset)
    dataset = pd.concat([dataset, new_dataset])
    dataset = dataset.drop(columns=["property_id", "name", "surrounding_properties"])

    #Reset index
    dataset["id"] = list(range(len(dataset)))
    dataset["id_buffer"] = dataset["id"]
    dataset = dataset.set_index("id_buffer")

    #Clear requirements
    dataset["requirements"] = dataset.apply(lambda row: [{k: v for k, v in r.items() if k not in ["construction_year", "accessibility_type", "accessibility_rating"]} for r in row["requirements"]], axis=1)

    print("Save dataset")
    dataset_dict = dataset.to_dict("list")
    saver = InversedJsonDatasetSaver(path="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/241025_data_pipeline/dataset.json")
    saver(dataset=dataset_dict)

if __name__ == "__main__":
    main()