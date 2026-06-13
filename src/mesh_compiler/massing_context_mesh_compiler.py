from typing import Any

import trimesh

from .mesh_compiler import MeshCompiler
from .massing_mesh_compiler import MassingMeshCompiler
from ..dataset_creator import DatasetCreator

class MassingContextMeshCompiler(MeshCompiler):
    def __init__(self, *, context_dataset_loader: DatasetCreator,
                       id_key: str,
                       context_ids_key: str,
                       massing_key: str) -> None:
        self.id_key = id_key
        self.context_ids_key = context_ids_key
        self.massing_key = massing_key
        self.massing_mesh_compiler = MassingMeshCompiler()
        context_dataset = context_dataset_loader()
        self.context_dataset = {
            context_dataset[self.id_key][i]: {
                key: value[i] for key, value in context_dataset.items()
            }
            for i in range(len(context_dataset[self.id_key]))
        }

    def __call__(self, *, obj: Any) -> trimesh.Trimesh:
        context_ids = obj[self.context_ids_key]
        massings = []
        for context_id in context_ids:
            if context_id not in self.context_dataset:
                continue
            for m in self.context_dataset[context_id][self.massing_key]:
                massing = {"id": m["id"], "massing": []}
                for extrusion in m["massing"]:
                    new_extrusion = {
                        "bottom_elevation": extrusion["bottom_elevation"],
                        "top_elevation": extrusion["top_elevation"],
                        "polygons": []
                    }
                    for polygon in extrusion["polygons"]:
                        new_extrusion["polygons"].append([
                            (round(p[0]), round(p[1]))
                            for p in polygon
                        ])
                    massing["massing"].append(new_extrusion)
                massings.append(massing)
        return self.massing_mesh_compiler(obj=massings)