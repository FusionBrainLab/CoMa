from typing import Dict, Any

from torch import nn
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from .model_creator import ModelCreator

class DCPModelLoader(ModelCreator):
    def __init__(self, *, base_model_creator: ModelCreator, 
                    path: str) -> None:
        self.base_model_creator = base_model_creator
        self.path = path
    
    def __call__(self) -> nn.Module:
        base_model = self.base_model_creator()
        reader = FileSystemReader(self.path)
        state_dict = get_model_state_dict(base_model)
        dcp.load(state_dict=state_dict, storage_reader=reader)
        return base_model