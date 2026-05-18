from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from .gradient_norm_computer import GradientNormComputer

class BaseGradientNormComputer(GradientNormComputer):
    norm_type: Any
    def __init__(self, *, norm_type: Any) -> None:
        self.norm_type = norm_type
    
    def __call__(self, *, model: nn.Module) -> float:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(grads, self.norm_type).item()
        return total_norm