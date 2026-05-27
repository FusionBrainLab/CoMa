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
import trimesh
from pandarallel import pandarallel
from diffusers import (
        QwenImageEditPipeline,
        QwenImageEditPlusPipeline
    )
import torch
from PIL import Image

import sys
sys.path.append("/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation")
from src.interpretable_function import InterpretableFunction
from src.core.function_utils import FunctionWrapper, FunctionGraph
from src.core.parsers import (
    DictToDictParser, AnyToDictParser, DictToAnyParser
)
from src.dataset_creator import (
    CsvDatasetCreator,
    PreprocessDatasetCreator,
    CoMaRegionsCreator,
    CoMaBuildingsCreator,
    InversedJsonChunkDatasetLoader
)
from src.dataset_processor import (
    GeoJSONToShapelyProcessor,
    GeoProjectProcessor,
    CompositionProcessor,
    FunctionProcessor,
    FootprintsToBuildingProcessor
)
from src.dataset_merger import GeoPandasDatasetIntersector, CoMaBuildingsRegionsMetadataMerger
from src.dataset_handler import InversedJsonChunkDatasetSaver
from src.mesh_compiler import MassingMeshCompiler, PolygonMeshCompiler
from src.image_data_visualizer import FocusMeshVisualizer, MapRegionVisualizer

def main():
    pandarallel.initialize(progress_bar=True, nb_workers=5)
    tqdm.pandas()
    """
    Order:
    131025_total_geo_graph_resaving
    171025_data_pipeline
    241025_data_pipeline
    231025_images_sampling
    261025_images_rendering
    271025_final_data_processing
    """
    #----------CREATE BUILDINGS----------
    print("----------CREATE BUILDINGS----------")

    buildings_creator = CoMaBuildingsCreator(
        footprints_dataset_path="/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/data_processing/raw_data/melbourne_data/2023-building-footprints.csv",
        building_id_col="structure_id",
        time_col="date_captured"
    )
    buildings = buildings_creator()

    #----------CREATE SITES----------
    print("----------CREATE SITES----------")

    regions_creator = CoMaRegionsCreator(
        properties_dataset_path="/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/data_processing/raw_data/melbourne_data/property-boundaries.csv",
        id_col="Gis_ID",
        output_col="site_contour"
    )
    regions = regions_creator()

    #----------CREATE MASSINGS----------
    print("----------CREATE MASSINGS----------")

    massings_creator = CoMaBuildingsRegionsMetadataMerger(
        buildings_dataset_key="buildings",
        regions_dataset_key="regions",
        buildings_col="building",
        regions_col="site_contour",
        metadata_path="/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/data_processing/raw_data/melbourne_data/buildings-with-name-age-size-accessibility-and-bicycle-facilities.csv",
        metadata_property_col="Property ID",
        metadata_time_col="Census year",
        metadata_feature_cols=["Predominant space use"],
        metadata_features_renaming={"Predominant space use":"function"},
        footprint_region_geo_match_threshold=0.9,
        features_col="requirements",
        massing_col="massing",
        context_radiuses=[50, 100, 200, 500]
    )
    massings = massings_creator(datasets={"buildings":buildings, "regions":regions})

    saver = InversedJsonChunkDatasetSaver(
        folder_path="/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/data_processing/coma/dataset/massings",
        chunk_length=10000
    )
    saver(dataset=massings)
    return

    loader = InversedJsonChunkDatasetLoader(
        folder_path="/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/data_processing/coma/dataset/massings",
        name_pattern=".*",
        verbose=True
    )
    massings = loader()

    #----------CREATE IMAGE CONTEXT----------
    print("----------CREATE IMAGE CONTEXT----------")

    massing_mesh_compiler = MassingMeshCompiler()
    site_mesh_compiler = PolygonMeshCompiler()

    massings = pd.DataFrame(massings)
    massings["id_buffer"] = massings["id"]
    massings = massings.set_index("id_buffer")

    def get_absolute_site(row):
        rel_site = row["site_contour"]
        base_point = row["base_point"]
        new_site = []
        for polygon in rel_site:
            new_p = [(round(p[0]+base_point[0], 2), round(p[1]+base_point[1], 2)) for p in polygon]
            new_site.append(new_p)
        return new_site
    #massings["absolute_site"] = massings.progress_apply(lambda row: get_absolute_site(row), axis=1)

    def get_massing_mesh(row):
        rel_massing = row["massing"]
        base_point = row["base_point"]
        new_massing = []
        for m in rel_massing:
            new_m = {"id":m["id"], "massing":[]}
            for e in m["massing"]:
                new_e = {"polygons":[]}
                for polygon in e["polygons"]:
                    new_p = [(round(p[0]+base_point[0], 2), round(p[1]+base_point[1], 2)) for p in polygon]
                    new_e["polygons"].append(new_p)
                new_e["bottom_elevation"] = e["bottom_elevation"]
                new_e["top_elevation"] = e["top_elevation"]
                new_m["massing"].append(new_e)
            new_massing.append(new_m)
        try:
            return massing_mesh_compiler(obj=new_massing)
        except:
            return None

    def get_site_mesh(row):
        rel_site = row["site_contour"]
        base_point = row["base_point"]
        new_site = []
        for polygon in rel_site:
            new_p = [(round(p[0]+base_point[0], 2), round(p[1]+base_point[1], 2)) for p in polygon]
            new_site.append(new_p)
        try:
            return site_mesh_compiler(obj=new_site)
        except:
            return None

    def get_context_mesh(row):
        context_ids = row["context_ids"]
        context_meshes = massings[massings["id"].isin(context_ids)]["massing_mesh"].values.tolist()
        context_meshes = [
            mesh for mesh in context_meshes
            if mesh is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0
        ]
        if len(context_meshes) == 0:
            return None
        mesh = trimesh.util.concatenate(context_meshes)
        return mesh

    #massings["massing_mesh"] = massings.progress_apply(lambda row: get_massing_mesh(row), axis=1)
    #massings["site_mesh"] = massings.progress_apply(lambda row: get_site_mesh(row), axis=1)

    #massings = massings[massings.progress_apply(lambda row: row["massing_mesh"] is not None and row["site_mesh"] is not None, axis=1)]

    def get_mesh_images(row):
        context_mesh = get_context_mesh(row)
        site_mesh = row["site_mesh"]
        if context_mesh is None:
            return

        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        base_folder = "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/data_processing/coma/dataset/context_images/mesh_images"
        os.makedirs(os.path.join(base_folder, row["id"]), exist_ok=True)
        for angle in angles:
            visualizer = FocusMeshVisualizer(
                focus_mesh_key="site_mesh",
                context_mesh_key="context_mesh",
                focus_mesh_color="red",
                context_mesh_color="grey",
                camera_distance_scale=7,
                vertical_angle=60,
                window_size=[1024, 1024],
                angle=angle
            )
            image = visualizer(data={"context_mesh":context_mesh, "site_mesh":site_mesh})
            image.save(os.path.join(base_folder, row["id"], f"{angle}.png"))
    #massings.progress_apply(lambda row: get_mesh_images(row), axis=1)

    def get_map_images(row):
        try:
            base_folder = "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/data_processing/coma/dataset/context_images/map_images"
            path = os.path.join(base_folder, f"{row['id']}.png")

            if os.path.exists(path):
                return

            region = row["absolute_site"]
            visualizer = MapRegionVisualizer(
                region_key="site_contour",
                fill_color="red",
                border_color="black",
                window_size=[1024, 1024],
                map_type="open-street-map",
                from_crs="EPSG:3857",
                zoom_level=14
            )
            image = visualizer(data={"site_contour":region})
            image.save(path)
        except:
            return
    #massings.progress_apply(get_map_images, axis=1)

    """pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
    pipeline.to("cuda")"""
    #pipeline.set_progress_bar_config(disable=True)
    prompt_1 = "Fill the white spaces with roads and parks, leaving no empty spaces. Leave red and grey objects untouched."
    prompt_2 = "Transform into a realistic urban landscape, preserving all contours, keep the red area unchanged."
    prompt_2 = "Photorealistic cityscape, ultra detailed, architectural photography, 8k\nTransform grey shapes into realistic buildings with (glass/concrete/steel) facades\nConvert roads to realistic asphalt with lane markings, parks to lush greenery\nMaintain exact contours and layout of all grey shapes and roads\nKeep red area completely untouched, solid red color, no rendering"

    n_prompt_2 = "Abstract, cartoon, painting, drawing, stylized, blurry\nDistorted architecture, mutated buildings, bad proportions\nModified layout, altered contours, changed footprints\nRendered red area, textured red zone, red buildings"
    base_inputs = {
        "image": None,
        "prompt": None,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 2.0,
        "num_inference_steps": 50,
        "num_images_per_prompt": 1,
        "height": 768,
        "width": 768
    }
    base_image_path = "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/data_processing/coma/dataset/context_images/mesh_images"
    output_image_path = "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/data_processing/coma/dataset/context_images/render_images"

    def get_render_images(row):
        local_base_path = os.path.join(base_image_path, row["id"])
        local_paths = [os.path.join(local_base_path, name) for name in os.listdir(local_base_path)]
        if len(local_paths) == 0:
            return
        base_images = [Image.open(local_path).convert("RGB").resize((768, 768), Image.LANCZOS) for local_path in local_paths]
        local_prompts = [prompt_1 for _ in range(len(base_images))]
        local_inputs = deepcopy(base_inputs)
        local_inputs["image"] = base_images
        local_inputs["prompt"] = local_prompts
        with torch.inference_mode():
            output = pipeline(**local_inputs)
            local_images = output.images
        del output
        del local_inputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        np_base_image = np.array(base_images[0])
        #local_images = [image.resize((np_base_image.shape[1], np_base_image.shape[0]), Image.LANCZOS) for image in local_images]
        
        local_inputs = deepcopy(base_inputs)
        local_inputs["image"] = local_images
        local_inputs["prompt"] = [prompt_2 for _ in range(len(local_images))]
        local_inputs["negative_prompt"] = [n_prompt_2 for _ in range(len(local_images))]
        with torch.inference_mode():
            output = pipeline(**local_inputs)
            local_images = output.images
        output_images = []
        for base_image, image in zip(base_images, local_images):
            np_image = np.array(image)
            base_image = base_image.resize((np_image.shape[1], np_image.shape[0]), Image.LANCZOS)
            base_np_image = np.array(base_image)
            red_lower = np.array([200, 0, 0])    # Minimum red threshold
            red_upper = np.array([255, 100, 100]) # Maximum red threshold
            red_mask = ((base_np_image >= red_lower) & (base_np_image <= red_upper)).all(axis=2)
            red_mask = np.repeat(red_mask[:, :, np.newaxis], 3, axis=2)

            np_image = np.where(red_mask, base_np_image, np_image)
            image = Image.fromarray(np_image)
            output_images.append(image)
        local_output_path = os.path.join(output_image_path, row["id"])
        os.makedirs(local_output_path, exist_ok=True)
        for i, name in enumerate(os.listdir(local_base_path)):
            output_images[i].save(os.path.join(local_output_path, name))
    #massings.progress_apply(get_render_images, axis=1)

if __name__ == "__main__":
    main()