from abc import ABC, abstractmethod
from typing import List, Any, Dict, Union, Tuple
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import time

import trimesh
import cadquery as cq

from .core.base import Function

def run_cad(cad, result_var, get_val):
    exec(cad, globals())
    compound = globals()[result_var]
    if get_val:
        compound = compound.val()
    vertices, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
    return mesh

class MassingCADExecutor(Function):
    def __init__(self, *, result_var: str,
                        get_val: bool,
                        export_folder: str) -> None:
        self.result_var = result_var
        self.get_val = get_val
        self.export_folder = export_folder

    def __call__(self, *, massing: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        exist_ids = os.listdir(self.export_folder)
        i = 0
        while any([e.startswith(f"{i}_") for e in exist_ids]):
            i += 1

        massing_mesh = []
        for m in massing:
            """with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                future = executor.submit(run_cad, m["massing"], self.result_var, self.get_val)
    
                start_time = time.time()
                while True:
                    if future.done():
                        mesh = future.result()
                        break

                    if time.time() - start_time > 2:
                        future.cancel()
                        raise TimeoutError("Process timeout")
                    
                    try:
                        pid = future._process.pid
                        process = psutil.Process(pid)
                        if process.memory_info().rss > 150 * 1024 * 1024:
                            process.kill()
                            raise MemoryError("Memory limit exceeded")
                    except:
                        pass
                    
                    time.sleep(0.1)"""
            
            mesh = run_cad(m["massing"], self.result_var, self.get_val)

            path = os.path.join(self.export_folder, f"{i}_{m['id']}.stl")
            mesh.export(path)
            massing_mesh.append({"id":m["id"], "massing":path})
        return massing_mesh