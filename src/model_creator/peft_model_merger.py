from typing import Dict, Any

from torch import nn

from .model_creator import ModelCreator

class PeftModelMerger(ModelCreator):
    def __init__(self, *, base_model_creator: ModelCreator) -> None:
        self.base_model_creator = base_model_creator
    
    def __call__(self) -> nn.Module:
        peft_model = self.base_model_creator()
        model = peft_model.merge_and_unload()
        return model