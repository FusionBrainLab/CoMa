from typing import Any, List, Union
import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class CheckpointModel(nn.Module):
    def __init__(self, *, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return checkpoint(self.base_model, *args, **kwargs)
