import json
import math
import os
import random
import io
from collections import deque
from copy import copy
import asyncio
from functools import partial
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

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
from pathos.multiprocessing import ProcessingPool
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("/workspace-SR008.fs2/maslov/massing_generation")
from src.dataset_handler import InversedJsonDatasetSaver
from src.dataset_creator import InversedJsonDatasetLoader, PreprocessDatasetCreator
from src.dataset_processor import (
    ContextualMassingVisualizer,
    FunctionProcessor
)
from src.core.function_utils import (
    FunctionWrapper
)
from src.interpretable_function import InterpretableFunction

def main():
    loader = InversedJsonDatasetLoader(path="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/271025_final_data_processing/dataset.json")
    visualizer = ContextualMassingVisualizer(
        property_dataset_creator=PreprocessDatasetCreator(
            base_creator=InversedJsonDatasetLoader(path="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/171025_data_pipeline/dataset.json"),
            preprocessor=FunctionProcessor(
                function=FunctionWrapper(
                    function=InterpretableFunction(
                        function="[{'id':m['id'], 'massing':[{'polygons':[[(point[0], point[1]) for point in polygon] for polygon in e['polygons']], 'bottom_elevation': e['bottom_elevation'], 'top_elevation': e['top_elevation']} for e in m['massing']]} for m in massing]",
                        returns_output=True
                    )
                ),
                cols_to_args_mapping={
                    "massing":"massing"
                },
                dataset_arg="dataset",
                output_key="massing"
            )
        ),
        property_id_col="id",
        property_massing_col="massing",
        base_id_col="id",
        base_massing_col="massing",
        images_folder="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/091125_dataset_overview/massing_images",
        images_col="massing_image",
        invalid_indicator_col="invalid_indicator"
    )
    dataset = loader()

    dataset = pd.DataFrame(dataset)
    dataset["id"] = dataset.apply(lambda row: str(row["id"]), axis=1)
    dataset["id_buffer"] = dataset["id"]
    dataset = dataset.set_index("id_buffer")
    dataset["massing"] = dataset.apply(lambda row: [{'id':m['id'], 'massing':[{'polygons':[[(row["base_point"][0] + point[0], row["base_point"][1] + point[1]) for point in polygon] for polygon in e['polygons']], 'bottom_elevation': e['bottom_elevation'], 'top_elevation': e['top_elevation']} for e in m['massing']]} for m in row["massing"]], axis=1)
    
    ind = 8041
    sample = dataset.loc[str(ind)].to_dict()
    sample_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/091125_dataset_overview/sample.json"
    with open(sample_path, "w+") as f:
        json.dump(sample, f)
    return

    #Save images
    #dataset = visualizer(dataset=dataset)

    #Get train-test statistics
    train_loader = InversedJsonDatasetLoader(path="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/071125_train_test_split/train_dataset.json")
    train_dataset = train_loader()
    test_loader = InversedJsonDatasetLoader(path="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/071125_train_test_split/test_dataset.json")
    test_dataset = test_loader()

    datasets = [train_dataset, test_dataset]
    plots_data = {
        "train":{

        },
        "test":{

        }
    }
    splits = ["train", "test"]
    for split, split_dataset in zip(splits, datasets):
        split_buildings = []
        def get_split_buildings(row):
            for r, m in zip(row["requirements"], row["massing"]):
                building = r
                building["id"] = f"{row["id"]}_{building["id"]}"
                building["massing"] = m["massing"]
                split_buildings.append(building)
        split_dataset = pd.DataFrame(split_dataset)
        split_dataset.apply(lambda row: get_split_buildings(row), axis=1)
        split_buildings = {k: [split_buildings[i][k] for i in range(len(split_buildings))] for k in split_buildings[0].keys()}
        split_buildings = pd.DataFrame(split_buildings)

        split_dataset["n_buildings"] = split_dataset.apply(lambda row: len(row["requirements"]), axis=1)
        plots_data[split]["Buildings per site"] = split_dataset["n_buildings"].values.tolist()

        plots_data[split]["Building functions"] = split_buildings["building_function"].values.tolist()
        plots_data[split]["Dwellings per building"] = split_buildings["n_dwellings"].values.tolist()
        plots_data[split]["Offices per building"] = split_buildings["n_commercial_spaces"].values.tolist()
        plots_data[split]["Floors per building"] = split_buildings["n_floors"].values.tolist()
        plots_data[split]["Building area"] = split_buildings["usable_area"].values.tolist()

        split_buildings["sum_public_capacity"] = split_buildings.apply(lambda row: sum([s["indoor_capacity"] + s["outdoor_capacity"] for s in row["public_spaces"]]), axis=1)
        plots_data[split]["Public capacity per building"] = split_buildings["sum_public_capacity"].values.tolist()

    plot_df_data = {}
    for name in plots_data["train"].keys():
        if name in ["Buildings per site"]:
            continue
        plot_df_data[name] = []
    plot_df_data["Split"] = []
    for split in plots_data.keys():
        length = None
        for key in plots_data[split].keys():
            if key in ["Buildings per site"]:
                continue
            length = len(plots_data[split][key])
            data = plots_data[split][key]
            if key in ["Dwellings per building", "Offices per building", "Floors per building", "Public capacity per building"]:
                data = [s + 0.1 for s in data]
            plot_df_data[key].extend(data)
        plot_df_data["Split"].extend([split] * length)
    plot_df = pd.DataFrame(plot_df_data)
    fontsize=14
    local_fontsize = 22
    sns.set_style("whitegrid") 
    for name in plots_data["train"].keys():
        if name in ["Building functions", "Buildings per site"]:
            continue
        train_data = plots_data["train"][name]
        test_data = plots_data["test"][name]

        plt.figure(figsize=(10, 6))

        ax = sns.displot(data=plot_df, x=name, hue='Split', kind='kde', 
            fill=True, common_norm=False, palette=['blue', 'orange'],
            alpha=0.2, height=6, aspect=1.5, log_scale=True)

        plt.xlabel('Values', fontsize=local_fontsize)
        plt.ylabel('Density', fontsize=local_fontsize)
        ax.tick_params(axis='x', labelsize=local_fontsize)
        ax.tick_params(axis='y', labelsize=local_fontsize)
        plt.setp(ax.legend.get_title(), fontsize=local_fontsize)
        plt.setp(ax.legend.get_texts(), fontsize=local_fontsize)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        folder = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/091125_dataset_overview/split_plots"
        path = os.path.join(folder, f"{name}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
    
    total_functions_data = plots_data["train"]["Building functions"] + plots_data["test"]["Building functions"]
    functions_df = pd.DataFrame({"Building functions":total_functions_data})
    
    value_counts = functions_df['Building functions'].value_counts()

    #Building count plot
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data=plots_data["train"]["Buildings per site"], stat='density', color='blue', alpha=0.2, 
             label='Train', bins=30)
    ax.tick_params(axis='x', labelsize=local_fontsize)
    ax.tick_params(axis='y', labelsize=local_fontsize)
    ax = sns.histplot(data=plots_data["test"]["Buildings per site"], stat='density', color='orange', alpha=0.2, 
             label='Test', bins=30)
    ax.tick_params(axis='x', labelsize=local_fontsize)
    ax.tick_params(axis='y', labelsize=local_fontsize)
    
    plt.xlabel('Density', fontsize=local_fontsize)
    plt.ylabel('Values', fontsize=local_fontsize)
    plt.legend(fontsize=local_fontsize)
    plt.tight_layout()
    path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/091125_dataset_overview/split_plots/Building per site.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')

    # Functions plot
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(x=value_counts.values, y=value_counts.index, palette='viridis')

    plt.xlabel('Count', fontsize=fontsize)
    plt.ylabel('Values', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/091125_dataset_overview/split_plots/Building functions.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()