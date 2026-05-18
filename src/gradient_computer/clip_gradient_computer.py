from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
from torch import nn

from .gradient_computer import GradientComputer
from ..gradient_norm_computer import GradientNormComputer

class ClipGradientComputer(GradientComputer):
    base_gradient_computer: GradientComputer
    gradient_norm_computer: GradientNormComputer
    max_norm: float
    last_norm: float
    def __init__(self, *, base_gradient_computer: GradientComputer,
                    gradient_norm_computer: GradientNormComputer,
                    max_norm: float) -> None:
        self.base_gradient_computer = base_gradient_computer
        self.gradient_norm_computer = gradient_norm_computer
        self.max_norm = max_norm
        self.last_norm = 0
    
    def __call__(self, *, model: nn.Module, batch: Dict[str, torch.Tensor]) -> None:
        self.base_gradient_computer(model=model, batch=batch)
        total_norm = self.gradient_norm_computer(model=model)
        torch.nn.utils.clip_grads_with_norm_(model.parameters(), max_norm=self.max_norm, total_norm=torch.tensor(total_norm))
        self.last_norm = self.gradient_norm_computer(model=model)