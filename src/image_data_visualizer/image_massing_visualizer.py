from abc import ABC, abstractmethod
from typing import Any, Dict
import base64
from io import BytesIO
import math

from PIL import Image
import trimesh
import numpy as np
import pyvista as pv

from .image_data_visualizer import ImageDataVisualizer
from ..massing_to_trimesh_converter import MassingToTrimeshConverter

class ImageMassingVisualizer(ImageDataVisualizer):
    def __init__(self, *, massing_key: str) -> None:
        self.massing_key = massing_key
        self.mesh_creator = MassingToTrimeshConverter()

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        massing = data[self.massing_key]
        meshes = [self.mesh_creator(massing=sample["massing"]) for sample in massing]
        mesh = meshes[0]
        for i in range(1, len(meshes)):
            try:
                mesh = mesh.union(meshes[i])
            except:
                mesh = trimesh.utils.concatenate([mesh, meshes[i]])
        
        with pv.vtk_verbosity('off'):
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(
                mesh, 
                color="red", 
                opacity=1,
                show_edges=False
            )
            image = plotter.screenshot(return_img=True)
        return image