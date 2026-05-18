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
from diffusers import (
        QwenImageEditPipeline,
        QwenImageEditPlusPipeline
    )
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sys
sys.path.append("/workspace-SR008.fs2/maslov/massing_generation")
from src.dataset_creator import InversedJsonDatasetLoader

def inference(rank, world_size, dataset):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    rank_inds = [[j for j in range(len(dataset)) if j % world_size == i] for i in range(world_size)]
    rank_datasets = [dataset.iloc[inds] for inds in rank_inds]
    dataset = rank_datasets[rank]

    images_folder = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/231025_images_sampling/env_image"
    low_folder = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/261025_images_rendering/env_renders_low"
    high_folder = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/261025_images_rendering/env_renders_high"
    pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
    pipeline.to(rank)
    pipeline.set_progress_bar_config(disable=True)

    iterator = dataset.iterrows()
    if rank == 0:
        iterator = tqdm(dataset.iterrows(), total=len(dataset))

    for ind, row in iterator:
        row_id = row["id"]
        image = Image.open(os.path.join(images_folder, f"{row_id}.png")).convert("RGB")

        base_image = image

        inputs = {
            "image": image,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 2.0,
            "negative_prompt": " ",
            "num_inference_steps": 15,
        }
        prompt_1 = "Fill the white spaces with roads and parks, leaving no empty spaces. Leave red and grey objects untouched."
        prompt_2 = "Transform into a realistic urban landscape, preserving all contours, keep the red area unchanged."
        #prompt_2 = "Transform gray figures into realistic buildings while preserving contours, make the landscape realistic without empty space, keep the red area unchanged."
        #prompt_2 = "Transform into a realistic cityscape by turning gray objects into buildings, keep the red area unchanged."

        #prompt_2 = "Photorealistic cityscape, ultra detailed, architectural photography, 8k\nTransform grey shapes into realistic buildings with (glass/concrete/steel) facades\nConvert roads to realistic asphalt with lane markings, parks to lush greenery\nMaintain exact contours and layout of all grey shapes and roads\nKeep red area completely untouched, solid red color, no rendering"
        n_prompt_2 = "Abstract, cartoon, painting, drawing, stylized, blurry\nDistorted architecture, mutated buildings, bad proportions\nModified layout, altered contours, changed footprints\nRendered red area, textured red zone, red buildings"

        inputs["prompt"] = prompt_1
        with torch.inference_mode():
            output = pipeline(**inputs)
            image = output.images[0]
        
        path = os.path.join(low_folder, f"{row_id}.png")
        image.save(path)

        inputs["prompt"] = prompt_2
        inputs["negative_prompt"] = n_prompt_2
        inputs["image"] = image
        with torch.inference_mode():
            output = pipeline(**inputs)
            image = output.images[0]
        
        np_image = np.array(image)
        base_image = base_image.resize((np_image.shape[1], np_image.shape[0]), Image.LANCZOS)
        base_np_image = np.array(base_image)
        red_lower = np.array([200, 0, 0])    # Minimum red threshold
        red_upper = np.array([255, 100, 100]) # Maximum red threshold
        red_mask = ((base_np_image >= red_lower) & (base_np_image <= red_upper)).all(axis=2)
        red_mask = np.repeat(red_mask[:, :, np.newaxis], 3, axis=2)

        np_image = np.where(red_mask, base_np_image, np_image)
        image = Image.fromarray(np_image)

        path = os.path.join(high_folder, f"{row_id}.png")
        image.save(path)

def main():
    loader = InversedJsonDatasetLoader(path="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/241025_data_pipeline/dataset.json")
    dataset = loader()
    dataset = pd.DataFrame(dataset)
    dataset["id"] = dataset.apply(lambda row: str(row["id"]), axis=1)
    dataset["id_buffer"] = dataset["id"]
    dataset = dataset.set_index("id_buffer")

    folders = ["env_image", "env_mesh", "map_image_alt", "map_image_base", "map_image_env"]
    ids = set([n.split(".")[0] for n in os.listdir(f"/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/231025_images_sampling/{folders[0]}")])
    for i in range(1, len(folders)):
        path = f"/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/231025_images_sampling/{folders[i]}"
        local_ids = set([n.split(".")[0] for n in os.listdir(path)])
        ids = ids.intersection(local_ids)
    
    dataset = dataset.loc[list(ids)]

    world_size = 8
    rank_inds = [[j for j in range(len(dataset)) if j % world_size == i] for i in range(world_size)]
    rank_datasets = [dataset.iloc[inds] for inds in rank_inds]
    mp.spawn(inference, args=(world_size, dataset), nprocs=world_size, join=True)
    
    """images_folder = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/231025_images_sampling/env_image"
    low_folder = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/261025_images_rendering/env_renders_low"
    high_folder = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/261025_images_rendering/env_renders_high"
    pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", device_map="balanced", torch_dtype=torch.bfloat16)
    pipeline.set_progress_bar_config(disable=True)"""

    """b_size = 4
    batches = []
    cur_batch = []
    for i, sample in enumerate(dataset.iterrows()):
        ind, row = sample
        cur_batch.append(row["id"])
        if (i + 1) % b_size == 0 or i == len(dataset) - 1:
            batches.append(cur_batch)
            cur_batch = []
        
    for b in tqdm(batches):
        images = [Image.open(os.path.join(images_folder, f"{row_id}.png")).convert("RGB") for row_id in b]
        base_images = images

        inputs = {
            "image": images,
            "generator": [torch.manual_seed(0)] * len(b),
            "true_cfg_scale": 2.0,
            "negative_prompt": [" "] * len(b),
            "num_inference_steps": 50,
            "num_images_per_prompt": 1
        }

        prompt_1 = "Fill the white spaces with roads and parks, leaving no empty spaces. Leave red and grey objects untouched."
        prompt_2 = "Transform into a realistic urban landscape, preserving all contours, keep the red area unchanged."
        n_prompt_2 = "Abstract, cartoon, painting, drawing, stylized, blurry\nDistorted architecture, mutated buildings, bad proportions\nModified layout, altered contours, changed footprints\nRendered red area, textured red zone, red buildings"

        inputs["prompt"] = [prompt_1] * len(b)
        with torch.inference_mode():
            output = pipeline(**inputs)
            images = output.images

        for i, image in enumerate(images):
            path = os.path.join(output_folder, f"{b[i]}_1.png")
            image.save(path)

        inputs["prompt"] = [prompt_2] * len(b)
        inputs["negative_prompt"] = [n_prompt_2] * len(b)
        inputs["image"] = images
        with torch.inference_mode():
            output = pipeline(**inputs)
            images = output.images
        
        for i, image in enumerate(images):
            np_image = np.array(image)
            base_image = base_images[i]
            base_image = base_image.resize((np_image.shape[1], np_image.shape[0]), Image.LANCZOS)
            base_np_image = np.array(base_image)
            red_lower = np.array([200, 0, 0])    # Minimum red threshold
            red_upper = np.array([255, 100, 100]) # Maximum red threshold
            red_mask = ((base_np_image >= red_lower) & (base_np_image <= red_upper)).all(axis=2)
            red_mask = np.repeat(red_mask[:, :, np.newaxis], 3, axis=2)

            np_image = np.where(red_mask, base_np_image, np_image)
            image = Image.fromarray(np_image)

            path = os.path.join(output_folder, f"{b[i]}_2.png")
            image.save(path)"""

    """for ind, row in tqdm(dataset.iterrows(), total=len(dataset)):
        row_id = row["id"]
        image = Image.open(os.path.join(images_folder, f"{row_id}.png")).convert("RGB")

        base_image = image

        inputs = {
            "image": image,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 2.0,
            "negative_prompt": " ",
            "num_inference_steps": 15,
        }
        prompt_1 = "Fill the white spaces with roads and parks, leaving no empty spaces. Leave red and grey objects untouched."
        prompt_2 = "Transform into a realistic urban landscape, preserving all contours, keep the red area unchanged."
        #prompt_2 = "Transform gray figures into realistic buildings while preserving contours, make the landscape realistic without empty space, keep the red area unchanged."
        #prompt_2 = "Transform into a realistic cityscape by turning gray objects into buildings, keep the red area unchanged."

        #prompt_2 = "Photorealistic cityscape, ultra detailed, architectural photography, 8k\nTransform grey shapes into realistic buildings with (glass/concrete/steel) facades\nConvert roads to realistic asphalt with lane markings, parks to lush greenery\nMaintain exact contours and layout of all grey shapes and roads\nKeep red area completely untouched, solid red color, no rendering"
        n_prompt_2 = "Abstract, cartoon, painting, drawing, stylized, blurry\nDistorted architecture, mutated buildings, bad proportions\nModified layout, altered contours, changed footprints\nRendered red area, textured red zone, red buildings"

        inputs["prompt"] = prompt_1
        with torch.inference_mode():
            output = pipeline(**inputs)
            image = output.images[0]
        
        path = os.path.join(low_folder, f"{row_id}.png")
        image.save(path)

        inputs["prompt"] = prompt_2
        inputs["negative_prompt"] = n_prompt_2
        inputs["image"] = image
        with torch.inference_mode():
            output = pipeline(**inputs)
            image = output.images[0]
        
        np_image = np.array(image)
        base_image = base_image.resize((np_image.shape[1], np_image.shape[0]), Image.LANCZOS)
        base_np_image = np.array(base_image)
        red_lower = np.array([200, 0, 0])    # Minimum red threshold
        red_upper = np.array([255, 100, 100]) # Maximum red threshold
        red_mask = ((base_np_image >= red_lower) & (base_np_image <= red_upper)).all(axis=2)
        red_mask = np.repeat(red_mask[:, :, np.newaxis], 3, axis=2)

        np_image = np.where(red_mask, base_np_image, np_image)
        image = Image.fromarray(np_image)

        path = os.path.join(high_folder, f"{row_id}.png")
        image.save(path)"""

if __name__ == "__main__":
    main()