from typing import Dict, Any

from torch import nn
import torch

from .model_creator import ModelCreator

class CompileModelCreator(ModelCreator):
    def __init__(self, *, base_model_creator: ModelCreator) -> None:
        self.base_model_creator = base_model_creator
    
    def __call__(self) -> nn.Module:
        model = self.base_model_creator()
        model = torch.compile(model)
        return model