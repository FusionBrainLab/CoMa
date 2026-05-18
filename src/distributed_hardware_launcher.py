from typing import Any, Dict, List, Union, Protocol, runtime_checkable
from dataclasses import dataclass
import os

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from .core.base import Function
from .core.attr_access import AttrGetter

@dataclass
class ParallelizeConfig:
    dim: str
    module_paths: List[List[Union[str, int]]]

@runtime_checkable
class Parallelizable(Protocol):
    def parallelize(self, *, device_mesh: DeviceMesh) -> None:
        ...

class DistributedHardwareLauncher(Function):
    module: object
    device_mesh_dict: Dict[str, int]
    parallelize_configs: List[ParallelizeConfig]
    local_rank: int
    device: torch.device
    device_mesh: DeviceMesh
    def __init__(self, *, module: object,
                 backend: str,
                 device_mesh_dict: Dict[str, int],
                 parallelize_configs: List[ParallelizeConfig]) -> None:
        self.module = module
        self.backend = backend
        self.device_mesh_dict = device_mesh_dict
        self.parallelize_configs = parallelize_configs
    
    def __call__(self, **kwargs: Any) -> Any:
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        
        dims = []
        names = []
        for dim, size in self.device_mesh_dict.items():
            dims.insert(0, size)
            names.insert(0, dim)
        torch.distributed.init_process_group(self.backend)
        self.device_mesh = init_device_mesh("cuda", dims, mesh_dim_names=tuple(names))
        
        attr_getter = AttrGetter()
        for config in self.parallelize_configs:
            devices = self.device_mesh[config.dim]
            for module_path in config.module_paths:
                if module_path[0] == "MODULE":
                    submodule = attr_getter(self.module, module_path[1:]) if len(module_path) > 1 else self.module
                elif module_path[0] == "KWARGS":
                    submodule = attr_getter(kwargs, module_path[1:])
                submodule.parallelize(device_mesh=devices)
        return self.module(**kwargs)
