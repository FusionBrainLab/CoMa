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

class FocusMeshVisualizer(ImageDataVisualizer):
    def __init__(self, *, focus_mesh_key: str,
                        context_mesh_key: str,
                        focus_mesh_color: str,
                        context_mesh_color: str,
                        window_size: List[int],
                        angle: int,
                        camera_distance_scale: float,
                        vertical_angle: int) -> None:
        self.focus_mesh_key = focus_mesh_key
        self.context_mesh_key = context_mesh_key
        self.focus_mesh_color = focus_mesh_color
        self.context_mesh_color = context_mesh_color
        self.window_size = window_size
        self.angle = angle
        self.camera_distance_scale = camera_distance_scale
        self.vertical_angle = vertical_angle

        pv.global_theme.allow_empty_mesh = True

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        focus_mesh = data[self.focus_mesh_key]
        context_mesh = data[self.context_mesh_key]

        with pv.vtk_verbosity('off'):
            focus_vertices = focus_mesh.vertices
            x_min, x_max = np.min(focus_vertices[:, 0]), np.max(focus_vertices[:, 0])
            y_min, y_max = np.min(focus_vertices[:, 1]), np.max(focus_vertices[:, 1])
            z_min, z_max = np.min(focus_vertices[:, 2]), np.max(focus_vertices[:, 2])
            focus_width = x_max - x_min
            focus_length = y_max - y_min
            focus_height = z_max - z_min
            radius = self.camera_distance_scale*max(focus_width, focus_length, focus_height)
            focus_center = np.array([
                (np.max(focus_vertices[:, 0]) + np.min(focus_vertices[:, 0]))/2, 
                (np.max(focus_vertices[:, 1]) + np.min(focus_vertices[:, 1]))/2, 
                (np.max(focus_vertices[:, 2]) + np.min(focus_vertices[:, 2]))/2
            ])

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(
                focus_mesh, 
                color=self.focus_mesh_color, 
                opacity=1,
                show_edges=False
            )
            plotter.add_mesh(
                context_mesh, 
                color=self.context_mesh_color, 
                opacity=1,
                show_edges=False
            )
            plotter.reset_camera()

            custom_camera_position = (
                (
                    focus_center[0] + radius*math.cos(math.radians(self.angle)),
                    focus_center[1] + radius*math.sin(math.radians(self.angle)),
                    focus_center[2] + radius*math.sin(math.radians(self.vertical_angle))
                ), 
                focus_center, 
                (0.0, 0, 1)
            )
            
            plotter.camera_position = custom_camera_position
            image = plotter.screenshot(return_img=True)

            np_image = np.array(image)
            image = Image.fromarray(np_image)
            return image
