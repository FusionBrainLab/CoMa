from typing import Dict, Any

from torch import nn

from .model_creator import ModelCreator
from ..torch_nn_module import *

class DefaultModelCreator(ModelCreator):
    model_cls: str
    model_args: Dict[str, Any]
    def __init__(self, *, model_cls: str, model_args: Dict[str, Any]) -> None:
        self.model_cls = model_cls
        self.model_args = model_args
    
    def __call__(self) -> nn.Module:
        model = globals()[self.model_cls](**self.model_args)
        return model