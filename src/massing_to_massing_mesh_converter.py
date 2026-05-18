from abc import ABC, abstractmethod
from typing import List, Any, Dict, Union, Tuple
import os

import numpy as np

from .core.base import Function
from .massing_to_trimesh_converter import MassingToTrimeshConverter

class MassingToMassingMeshConverter(Function):
    def __init__(self, *, meshes_folder: str,
                        export_mesh: bool) -> None:
        self.meshes_folder = meshes_folder
        self.export_mesh = export_mesh
        self.mesh_creator = MassingToTrimeshConverter()

    def __call__(self, *, massing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        exist_ids = os.listdir(self.meshes_folder)
        i = 0
        while any([e.startswith(f"{i}_") for e in exist_ids]):
            i += 1

        massing_mesh = []
        for m in massing:
            mesh = self.mesh_creator(massing=m["massing"])
            if self.export_mesh:
                path = os.path.join(self.meshes_folder, f"{i}_{m['id']}.stl")
                mesh.export(path)
                local_mesh = path
            else:
                local_mesh = mesh
            massing_mesh.append({"id":m["id"], "massing":local_mesh})
        return massing_mesh
