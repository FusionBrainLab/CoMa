from abc import ABC, abstractmethod
from typing import Dict, Any, Union, TypedDict

import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from transformers import get_cosine_schedule_with_warmup

from .scheduled_optimizer import ScheduledOptimizer
from ..core.inheritance_utils import PrePostDecorator, PreMethodConfig

class CheckpointableScheduledOptimizer(ScheduledOptimizer):
    __call__ = PrePostDecorator(pre_method_config=PreMethodConfig(method_name="precall", update_input=False), post_method_config=None)
    
    def precall(self, *, model: nn.Module) -> None:
        if self.optimizer == None and getattr(self, "checkpoint", None) != None:
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = globals()[self.optimizer_cls](parameters, **self.optimizer_args)
            scheduler = globals()[self.scheduler_cls](optimizer, **self.scheduler_args)
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.model = model
            self.checkpoint["optimizer"].set_state(statefuls={"model":self.model, "optimizer":self.optimizer})
            self.checkpoint["scheduler"].set_state(statefuls={"scheduler":self.scheduler})
    
    def load_checkpoint(self, *, checkpoint: object) -> None:
        assert isinstance(checkpoint, dict)
        self.checkpoint = checkpoint
    
    def save_checkpoint(self) -> object:
        checkpoint = {}
        checkpoint["optimizer"] = {"model": self.model, "optimizer":self.optimizer}
        checkpoint["scheduler"] = {"scheduler":self.scheduler}
        return checkpoint