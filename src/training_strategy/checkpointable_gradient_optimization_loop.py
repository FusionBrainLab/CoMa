from abc import ABC, abstractmethod
from typing import Optional, TypedDict, Dict

from torch import nn
import json

from .gradient_optimization_loop import GradientOptimizationLoop
from ..core.inheritance_utils import PreMethodConfig, PrePostDecorator

class CheckpointableGradientOptimizationLoop(GradientOptimizationLoop):
    __call__ = PrePostDecorator(pre_method_config=PreMethodConfig(method_name="precall", update_input=False), post_method_config=None)
    
    def precall(self, *, model: nn.Module) -> None:
        model_checkpoint = getattr(self, "model_checkpoint", None)
        if model_checkpoint != None:
            model.load_checkpoint(checkpoint=self.model_checkpoint)
    
    def load_checkpoint(self, *, checkpoint: object) -> None:
        assert isinstance(checkpoint, dict)
        self.dataloader.load_checkpoint(checkpoint=checkpoint["dataloader"])
        self.gradient_optimizer.load_checkpoint(checkpoint=checkpoint["optimizer"])
        if self.model != None:
            self.model.load_checkpoint(checkpoint=checkpoint["model"])
        else:
            self.model_checkpoint = checkpoint["model"]
        
    def save_checkpoint(self) -> object:
        checkpoint = {}
        checkpoint["dataloader"] = self.dataloader.save_checkpoint()
        checkpoint["model"] = self.model.save_checkpoint()
        checkpoint["optimizer"] = self.gradient_optimizer.save_checkpoint()
        return checkpoint