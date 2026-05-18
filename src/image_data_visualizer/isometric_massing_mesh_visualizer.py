from typing import List, Dict, Any
import re

import pyvista as pv
import trimesh
import numpy as np

from .image_data_visualizer import ImageDataVisualizer
from PIL import Image

class IsometricMassingMeshVisualizer(ImageDataVisualizer):
    def __init__(self, *, window_size: List[int],
                        color: str,
                        load_mesh: bool,
                        angle: int,
                        massing_key: str) -> None:
        self.window_size = window_size
        self.color = color
        self.load_mesh = load_mesh
        self.angle = angle
        self.massing_key = massing_key

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        massing = data[self.massing_key]
        def get_mesh(m):
            return trimesh.load(m["massing"]) if self.load_mesh else m["massing"]
        
        meshes = [get_mesh(m) for m in massing]
        massing_mesh = trimesh.util.concatenate(meshes)
        with pv.vtk_verbosity('off'):
            plotter = pv.Plotter(
                off_screen=True,
                window_size=self.window_size
            )
            plotter.add_mesh(
                massing_mesh, 
                color=self.color, 
                opacity=1,
                show_edges=False
            )

            plotter.view_isometric()
            plotter.camera.azimuth = self.angle
            plotter.camera.up = (0, 0, 1)
            plotter.reset_camera_clipping_range()
            image = plotter.screenshot(return_img=True)

            np_image = np.array(image)
            image = Image.fromarray(np_image)
            return image