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
from src.interpretable_function import InterpretableFunction
from src.massing_to_trimesh_converter import MassingToTrimeshConverter
from src.dataset_creator import InversedJsonDatasetLoader
from src.polygons_to_shapely_converter import PolygonsToShapelyConverter
from src.polygons_map_trace_creator import PolygonsMapTraceCreator
from src.shapely_to_polygons_converter import ShapelyToPolygonsConverter

def main():
    loader = InversedJsonDatasetLoader(path="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/171025_data_pipeline/dataset.json")
    base_dataset = loader()
    base_dataset = pd.DataFrame(base_dataset)
    base_dataset["id_buffer"] = base_dataset["id"]
    base_dataset = base_dataset.set_index("id_buffer")

    polygons_converter = PolygonsToShapelyConverter()
    shapely_converter = ShapelyToPolygonsConverter()
    map_visualizer = PolygonsMapTraceCreator(fill_color="red", line_color="black")

    loader = InversedJsonDatasetLoader(path="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/241025_data_pipeline/dataset.json")
    dataset = loader()
    dataset = pd.DataFrame(dataset)
    dataset["id_buffer"] = dataset["id"]
    dataset = dataset.set_index("id_buffer")
    print(len(dataset))

    public_samples = dataset[dataset.apply(lambda row: len([r for r in row["requirements"] if len(r["public_spaces"]) > 0]) > 0, axis=1)]
    office_samples = dataset[dataset.apply(lambda row: len([r for r in row["requirements"] if r["n_commercial_spaces"] > 0]) > 0, axis=1)]
    dwelling_samples = dataset[dataset.apply(lambda row: len([r for r in row["requirements"] if r["n_dwellings"] > 0]) > 0, axis=1)]
    common_samples = dataset[(dataset["id"].isin(public_samples.index)) & (dataset["id"].isin(office_samples.index)) & (dataset["id"].isin(dwelling_samples.index))]
    print(len(public_samples), len(office_samples), len(dwelling_samples), len(common_samples))

    #Visualize common sample
    ind = random.randint(0, len(common_samples))
    sample = common_samples.iloc[ind].to_dict()

    mesh_creator = MassingToTrimeshConverter()
    def get_mesh(row):
        meshes = [mesh_creator(massing=sample["massing"]) for sample in row["massing"]]
        mesh = trimesh.util.concatenate(meshes)
        return mesh
    sample["mesh"] = get_mesh(sample)

    def get_env_mesh(row):
        env = base_dataset[base_dataset["id"].isin(row["env_properties"])]
        env_massings = env["massing"].values.tolist()
        env_meshes = [get_mesh({"massing":m}) for m in env_massings]
        mesh = trimesh.util.concatenate(env_meshes)
        return mesh
    sample["env_mesh"] = get_env_mesh(sample)

    def get_env_image(row):
        surrounding_mesh = row["env_mesh"]
        
        site = [[(p[0], p[1]) for p in polygon] for polygon in row["global_site_contour"]]
        site = polygons_converter(polygons=site)
        exterior_coords = list(site.exterior.coords)
        if exterior_coords[0] == exterior_coords[-1]:
            exterior_coords = exterior_coords[:-1]
        holes_coords = []
        for interior in site.interiors:
            interior_coords = list(interior.coords)
            if interior_coords[0] == interior_coords[-1]:
                interior_coords = interior_coords[:-1]
            holes_coords.append(interior_coords)
        exterior_array = np.array(exterior_coords)
        holes_arrays = [np.array(hole) for hole in holes_coords]
        poly_2d = trimesh.path.polygons.Polygon(exterior_array, holes=holes_arrays)
        site_mesh = trimesh.creation.extrude_polygon(poly_2d, height=0.5)

        site_vertices = site_mesh.vertices
        x_min, x_max = np.min(site_vertices[:, 0]), np.max(site_vertices[:, 0])
        y_min, y_max = np.min(site_vertices[:, 1]), np.max(site_vertices[:, 1])
        site_width = x_max - x_min
        site_length = y_max - y_min
        radius = 5*max(site_width, site_length)
        site_center = np.array([
            (np.max(site_vertices[:, 0]) + np.min(site_vertices[:, 0]))/2, 
            (np.max(site_vertices[:, 1]) + np.min(site_vertices[:, 1]))/2, 
            (np.max(site_vertices[:, 2]) + np.min(site_vertices[:, 2]))/2
        ])

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(
            site_mesh, 
            color="red", 
            opacity=1,
            show_edges=False
        )
        plotter.add_mesh(
            surrounding_mesh, 
            color="grey", 
            opacity=1,
            show_edges=False
        )
        plotter.reset_camera()

        xs = [0, -1, 0, 1]
        ys = [-1, 0, 1, 0]

        max_perc = 0
        max_ind = 0
        max_image = None
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            custom_camera_position = (
                (
                    site_center[0] + x*radius*math.cos(math.pi/4),
                    site_center[1] + y*radius*math.cos(math.pi/4),
                    site_center[2] + radius*math.sin(math.pi/4)
                ), 
                site_center, 
                (0.0, 0, 1)
            )
            
            plotter.set_viewup(custom_camera_position[2])
            plotter.set_position(custom_camera_position[0])
            plotter.set_focus(custom_camera_position[1])
            image = plotter.screenshot(return_img=True)
            
            np_image = np.array(image)
            image = Image.fromarray(np_image)

            red_lower = np.array([200, 0, 0])    # Minimum red threshold
            red_upper = np.array([255, 100, 100]) # Maximum red threshold
            red_mask = np.all((np_image >= red_lower) & (np_image <= red_upper), axis=-1)
            red_pixel_count = np.sum(red_mask)
            total_pixels = np_image.shape[0] * np_image.shape[1]
            red_percentage = red_pixel_count / total_pixels
            if red_percentage > max_perc:
                max_perc = red_percentage
                max_ind = i
                max_image = image
        return {"rotation": max_ind, "image":max_image}
    with pv.vtk_verbosity('off'):
        sample["env_image"] = get_env_image(sample)
    path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/201025_dataset_analysis/sample_env.png"
    sample["env_image"]["image"].save(path)

    def get_map_image(row):
        geo_site = [[(p[0], p[1]) for p in polygon] for polygon in row["geo_site_contour"]]
        sample_map = map_visualizer(polygons=geo_site)
        map_fig = go.Figure()
        for trace in sample_map:
            map_fig.add_trace(trace)

        total_lons = []
        total_lats = []
        for i, polygon in enumerate(geo_site):
            lons = [p[0] for p in polygon] + [polygon[0][0]]
            lats = [p[1] for p in polygon] + [polygon[0][1]]
            total_lons.extend(lons)
            total_lats.extend(lats)
        
        min_lon, max_lon = min(total_lons), max(total_lons)
        min_lat, max_lat = min(total_lats), max(total_lats)

        # Calculate zoom level based on polygon size
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        max_range = max(lon_range, lat_range)

        # Dynamic zoom calculation
        zoom_level = max(1, min(20, 14 - math.log(max_range * 100)))

        map_fig.update_layout(
            mapbox={
                'style': "open-street-map",
                'center': {'lon': sum(total_lons)/len(total_lons), 'lat': sum(total_lats)/len(total_lats)},
                "zoom":zoom_level
            },
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0}
        )
        map_fig.to_html()
        image = map_fig.to_image(format="png")
        image = Image.open(io.BytesIO(image))
        image = image.rotate(90*row["env_image"]["rotation"])
        return image
    sample["map_image"] = get_map_image(sample)
    path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/201025_dataset_analysis/sample_map.png"
    sample["map_image"].save(path)

    path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/201025_dataset_analysis/sample.json"
    sample_metadata = {k: v for k, v in sample.items() if k not in ["mesh", "env_mesh", "env_image", "map_image"]}
    with open(path, "w+") as f:
        json.dump(sample_metadata, f)

if __name__ == "__main__":
    main()