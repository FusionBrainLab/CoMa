from typing import Optional, TypedDict, Dict

from torch import nn

from .training_strategy import TrainingStrategy
from ..data_loader import DataLoader
from ..gradient_computer import GradientComputer
from ..gradient_optimizer import GradientOptimizer

class GradientOptimizationLoop(TrainingStrategy):
    model: nn.Module
    dataloader: DataLoader
    gradient_computer: GradientComputer
    gradient_optimizer: GradientOptimizer
    def __init__(self, *, dataloader: DataLoader, 
                 gradient_computer: GradientComputer, 
                 gradient_optimizer: GradientOptimizer) -> None:
        self.dataloader = dataloader
        self.gradient_computer = gradient_computer
        self.gradient_optimizer = gradient_optimizer
        self.model = None

    def __call__(self, *, model: nn.Module) -> nn.Module:
        self.model = model
        for batch in self.dataloader():
            self.gradient_computer(model=self.model, batch=batch)
            self.gradient_optimizer(model=self.model)
        return self.model