from typing import Dict, Any

from torch import nn
from peft import PeftModel

from .model_creator import ModelCreator

class PeftAdaptersLoader(ModelCreator):
    def __init__(self, *, base_model_creator: ModelCreator,
                        adapters_path: str) -> None:
        self.base_model_creator = base_model_creator
        self.adapters_path = adapters_path
    
    def __call__(self) -> nn.Module:
        base_model = self.base_model_creator()
        model = PeftModel.from_pretrained(base_model, self.adapters_path)
        model = model.merge_and_unload()
        return model