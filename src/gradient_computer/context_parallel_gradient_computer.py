from typing import Dict, Any, List

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental import context_parallel

from .gradient_computer import GradientComputer

class ContextParallelGradientComputer(GradientComputer):
    def __init__(self, *, base_computer: GradientComputer,
                        buffer_keys: List[str],
                        seq_dims: List[int]) -> None:
        self.base_computer = base_computer
        self.buffer_keys = buffer_keys
        self.seq_dims = seq_dims
        self.device_mesh = None
        
    def __call__(self, *, model: nn.Module, batch: Dict[str, torch.Tensor]) -> None:
        buffers = [batch[key] for key in self.buffer_keys]
        with context_parallel(self.device_mesh, buffers=buffers, buffer_seq_dims=self.seq_dims, no_restore_buffers=buffers):
            self.base_computer(model=model, batch=batch)
    
    def parallelize(self, *, device_mesh: DeviceMesh) -> None:
        self.device_mesh = device_mesh