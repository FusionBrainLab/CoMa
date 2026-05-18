from typing import List, Dict, Any
import re
import json

import pyvista as pv
import trimesh
from PIL import Image
import pandas as pd
from pandarallel import pandarallel
import numpy as np
import math

from .image_data_visualizer import ImageDataVisualizer
from ..dataset_creator import DatasetCreator

class ContextualMassingMeshVisualizer(ImageDataVisualizer):
    def __init__(self, *, context_matching_dataset_loader: DatasetCreator,
                        massing_col: str,
                        id_col: str,
                        context_ids_col: str,
                        massing_color: str,
                        context_color: str,
                        window_size: List[int],
                        massing_key: str) -> None:
        self.context_matching_dataset_loader = context_matching_dataset_loader
        self.massing_col = massing_col
        self.context_ids_col = context_ids_col
        self.massing_color = massing_color
        self.context_color = context_color
        self.window_size = window_size
        self.massing_key = massing_key

        context_matching_dataset = context_matching_dataset_loader()
        context_dataset = pd.DataFrame(context_matching_dataset)
        context_dataset["id_buffer"] = context_dataset["id"]
        context_dataset = context_dataset.set_index("id_buffer")

        """#pandarallel.initialize(nb_workers=96, progress_bar=False)
        def get_mesh(row):
            return trimesh.load(row[self.massing_col][0]["massing"])
        context_dataset["mesh"] = context_dataset.parallel_apply(get_mesh, axis=1)"""

        self.context_dataset = context_dataset

        pv.global_theme.allow_empty_mesh = True

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        massing = data[self.massing_key]
        meshes = [trimesh.load(m["massing"]) for m in massing]
        massing_mesh = trimesh.util.concatenate(meshes)

        massing_str = json.dumps(massing)
        context_ids = self.context_dataset[self.context_dataset.apply(lambda row: json.dumps(row[self.massing_col]) == massing_str, axis=1)][self.context_ids_col].iloc[0]

        context_massings = self.context_dataset.loc[context_ids][self.massing_col].values.tolist()
        context_meshes = [trimesh.load(m[0]["massing"]) for m in context_massings]
        context_mesh = trimesh.util.concatenate(context_meshes)

        with pv.vtk_verbosity('off'):
            massing_vertices = massing_mesh.vertices
            x_min, x_max = np.min(massing_vertices[:, 0]), np.max(massing_vertices[:, 0])
            y_min, y_max = np.min(massing_vertices[:, 1]), np.max(massing_vertices[:, 1])
            z_min, z_max = np.min(massing_vertices[:, 2]), np.max(massing_vertices[:, 2])
            massing_width = x_max - x_min
            massing_length = y_max - y_min
            massing_height = z_max - z_min
            radius = 5*max(massing_width, massing_length, massing_height)
            massing_center = np.array([
                (np.max(massing_vertices[:, 0]) + np.min(massing_vertices[:, 0]))/2, 
                (np.max(massing_vertices[:, 1]) + np.min(massing_vertices[:, 1]))/2, 
                (np.max(massing_vertices[:, 2]) + np.min(massing_vertices[:, 2]))/2
            ])

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(
                massing_mesh, 
                color=self.massing_color, 
                opacity=1,
                show_edges=False
            )
            plotter.add_mesh(
                context_mesh, 
                color=self.context_color, 
                opacity=1,
                show_edges=False
            )
            plotter.reset_camera()

            xs = [0, -1, 0, 1]
            ys = [-1, 0, 1, 0]

            max_perc = 0
            max_ind = 0
            max_image = None
            error_image = None
            for i in range(len(xs)):
                x = xs[i]
                y = ys[i]
                custom_camera_position = (
                    (
                        massing_center[0] + x*radius*math.cos(math.pi/4),
                        massing_center[1] + y*radius*math.cos(math.pi/4),
                        massing_center[2] + radius*math.sin(math.pi/4)
                    ), 
                    massing_center, 
                    (0.0, 0, 1)
                )
                
                plotter.set_viewup(custom_camera_position[2])
                plotter.set_position(custom_camera_position[0])
                plotter.set_focus(custom_camera_position[1])
                image = plotter.screenshot(return_img=True)
                
                np_image = np.array(image)
                image = Image.fromarray(np_image)
                error_image = image

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
            if max_image == None:
                max_image = error_image

        image = max_image
        return image