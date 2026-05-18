from abc import ABC, abstractmethod
from typing import Dict, Any, Union, TypedDict

import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from transformers import get_cosine_schedule_with_warmup

from .gradient_optimizer import GradientOptimizer

class ScheduledOptimizer(GradientOptimizer):
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    model: nn.Module
    last_lr: float
    def __init__(self, *, optimizer_cls: str, 
                 optimizer_args: Dict[str, Any], 
                 scheduler_cls: str, 
                 scheduler_args: Dict[str, Any]) -> None:
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        self.scheduler_cls = scheduler_cls
        self.optimizer = None
        self.scheduler = None
        self.model = None
    
    def __call__(self, *, model: nn.Module) -> None:
        if self.optimizer == None:
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = globals()[self.optimizer_cls](parameters, **self.optimizer_args)
            scheduler = globals()[self.scheduler_cls](optimizer, **self.scheduler_args)
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.model = model
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.last_lr = self.scheduler.get_last_lr()[0]