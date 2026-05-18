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

import sys
sys.path.append("/workspace-SR008.fs2/maslov/massing_generation")
from src.dataset_handler import InversedJsonDatasetSaver
from src.dataset_creator import InversedJsonDatasetLoader


def main():
    loader = InversedJsonDatasetLoader(path="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/241025_data_pipeline/dataset.json")
    dataset = loader()
    dataset = pd.DataFrame(dataset)
    dataset["id"] = dataset.apply(lambda row: str(row["id"]), axis=1)
    dataset["id_buffer"] = dataset["id"]
    dataset = dataset.set_index("id_buffer")

    tqdm.pandas()
    print(dataset.columns)

    #Normalize contour and massing
    dataset["base_point"] = dataset.apply(lambda row: row["global_site_contour"][0][0], axis=1)
    dataset["site_contour"] = dataset.apply(lambda row: [[(round(p[0] - row["base_point"][0], 3), round(p[1] - row["base_point"][1], 3)) for p in polygon] for polygon in row["global_site_contour"]], axis=1)
    def normalize_massing(row):
        new_massings = []
        for m in row["massing"]:
            new_extrusions = []
            for e in m["massing"]:
                new_e = {
                    "polygons":[[(round(p[0] - row["base_point"][0], 3), round(p[1] - row["base_point"][1], 3)) for p in polygon] for polygon in e["polygons"]],
                    "bottom_elevation":e["bottom_elevation"],
                    "top_elevation":e["top_elevation"]
                }
                new_extrusions.append(new_e)
            new_massing = {
                "id":m["id"],
                "massing":new_extrusions
            }
            new_massings.append(new_massing)
        return new_massings
    dataset["massing"] = dataset.apply(lambda row: normalize_massing(row), axis=1)
    
    #Drop extra cols
    dataset = dataset.drop(columns=["surrounding_properties", "geo_site_contour", "global_site_contour"])

    #Drop rows that have no images
    folders = [
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/261025_images_rendering/env_renders_high",
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/261025_images_rendering/env_renders_low",
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/231025_images_sampling/env_image",
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/231025_images_sampling/map_image_base",
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/231025_images_sampling/map_image_alt"
    ]
    ids = set([n.split(".")[0] for n in os.listdir(f"{folders[0]}")])
    for i in range(1, len(folders)):
        local_ids = set([n.split(".")[0] for n in os.listdir(folders[i])])
        ids = ids.intersection(local_ids)
    
    dataset = dataset.loc[list(ids)]

    #Add images cols
    keys = ["env_render_high", "env_render_low", "env_image", "map_image_base", "map_image_alt"]
    for key, folder in zip(keys, folders):
        dataset[key] = dataset.apply(lambda row: os.path.join(folder, f"{row['id']}.png"), axis=1)

    #Save dataset
    print(dataset.columns)
    print(len(dataset))
    saver = InversedJsonDatasetSaver(path="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/271025_final_data_processing/dataset.json")
    dataset = dataset.to_dict("list")
    saver(dataset=dataset)

if __name__ == "__main__":
    #asyncio.run(main())
    main()