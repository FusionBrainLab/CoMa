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
from src.dataset_creator import InversedJsonDatasetLoader
from src.polygons_map_trace_creator import PolygonsMapTraceCreator

def main():
    footprint_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/131025_total_geo_graph_resaving/footprint_dataset.json"
    
    block_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/131025_total_geo_graph_resaving/block_dataset.json"
    property_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/131025_total_geo_graph_resaving/property_dataset.json"
    
    dwelling_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/131025_total_geo_graph_resaving/dwelling_dataset.json"
    building_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/131025_total_geo_graph_resaving/building_dataset.json"
    address_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/131025_total_geo_graph_resaving/address_dataset.json"
    business_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/131025_total_geo_graph_resaving/business_dataset.json"
    cafe_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/131025_total_geo_graph_resaving/cafe_dataset.json"
    bar_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/131025_total_geo_graph_resaving/bar_dataset.json"

    output = {}
    tqdm.pandas()

    #Analyse self polygon-point relations
    polygon_datasets = {}
    paths = [footprint_dataset_path, block_dataset_path, property_dataset_path]
    keys = ["footprint", "block", "property"]
    for i, k in enumerate(keys):
        path = paths[i]
        loader = InversedJsonDatasetLoader(path=path)
        dataset_dict = loader()
        dataset = pd.DataFrame(dataset_dict)
        polygon_datasets[k] = dataset

    point_polygon_relations = {}
    for k, dataset in polygon_datasets.items():
        dataset["point_polygon"] = dataset.apply(lambda row: row["id"] in row[f"{k}_point_intersects_ids"], axis=1)
        point_polygon_relations[k] = dataset["point_polygon"].value_counts().to_dict()
    output["self_point_polygon_compatibility"] = point_polygon_relations

    #Analyse property graph
    property_dataset = polygon_datasets["property"]
    predicates = ["contains", "within", "touches", "overlaps", "covers", "covered_by", "crosses", "intersects"]
    keys = ["footprint", "block", "property"]
    property_geo_graph = {}
    for k in keys:
        for p in predicates:
            property_dataset["n"] = property_dataset.apply(lambda row: len(row[f"{k}_{p}_ids"]), axis=1)
            property_geo_graph[f"n_{k}_{p}"] = property_dataset["n"].value_counts().to_dict()
        property_dataset["n"] = property_dataset.apply(lambda row: len(row[f"{k}_point_intersects_ids"]), axis=1)
        property_geo_graph[f"n_{k}_point_intersects"] = property_dataset["n"].value_counts().to_dict()
    output["property_geo_graph"] = property_geo_graph

    #Load footprints
    footprint_dataset = polygon_datasets["footprint"]

    #Visualize geometry relations
    for p in predicates:
        property_subset = property_dataset[property_dataset.apply(lambda row: len(row[f"footprint_{p}_ids"]) > 0, axis=1)]
        if len(property_subset) == 0:
            continue
        ids = property_subset[f"footprint_{p}_ids"].iloc[0]

        sample_footprints = footprint_dataset[footprint_dataset["id"].isin(ids)]["geo_projection"].values.tolist()
        fig = go.Figure()
        trace_creator = PolygonsMapTraceCreator(fill_color="red", line_color="blue")
        for projection in sample_footprints:
            polygons = [[tuple(p) for p in poly] for poly in projection]
            traces = trace_creator(polygons=polygons)
            for t in traces:
                fig.add_trace(t)

        trace_creator = PolygonsMapTraceCreator(fill_color="green", line_color="blue")
        property_polygons = property_subset["geo_projection"].iloc[0]
        polygons = [[tuple(p) for p in poly] for poly in property_polygons]
        traces = trace_creator(polygons=polygons)
        for t in traces:
            fig.add_trace(t)
        
        fig.update_layout(
            mapbox={
                'style': "open-street-map"
            },
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0}
        )
        sample_path = f"/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/131025_total_graph_analysis/{p}.html"
        fig.write_html(sample_path, config={
            'displayModeBar': True,
            'scrollZoom': True
        })

    #Analyse structure-containing properties
    footprint_dataset = footprint_dataset[footprint_dataset["footprint_type"] == "Structure"]
    footprint_dataset["dated_structure_id"] = footprint_dataset.apply(lambda row: f"{row['structure_id']}_{row['date_captured']}", axis=1)
    structure_dataset = footprint_dataset.groupby('dated_structure_id').agg(list)
    structure_sets = [set(ids) for ids in structure_dataset['id']]
    
    def count_full_structures(row, pred_col):
        footprint_ids = set(row[pred_col])
        count = len([s for s in structure_sets if len(s.intersection(footprint_ids)) == len(s)])
        return count

    def count_partial_structures(row, pred_col):
        footprint_ids = set(row[pred_col])
        count = len([s for s in structure_sets if 0 < len(s.intersection(footprint_ids)) < len(s)])
        return count
    output["structure_covering"] = {}
    structure_predicates = predicates + ["point_intersects"]
    for p in structure_predicates:
        property_dataset["n_full_structures"] = property_dataset.progress_apply(lambda row: count_full_structures(row, f"footprint_{p}_ids"), axis=1)
        property_dataset["n_partial_structures"] = property_dataset.progress_apply(lambda row: count_partial_structures(row, f"footprint_{p}_ids"), axis=1)
        p_output = {}
        p_output["n_full_structures"] = property_dataset["n_full_structures"].value_counts().to_dict()
        p_output["n_partial_structures"] = property_dataset["n_partial_structures"].value_counts().to_dict()
        output["structure_covering"][p] = p_output

    with open("/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/131025_total_graph_analysis/output.json", "w+") as f:
        json.dump(output, f)

if __name__ == "__main__":
    main()