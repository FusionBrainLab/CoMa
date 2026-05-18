from typing import Dict, Any

from torch import nn
from peft import LoraModel, LoraConfig, get_peft_model

from .model_creator import ModelCreator

class LoRAModelCreator(ModelCreator):
    def __init__(self, *, base_model_creator: ModelCreator,
                        lora_config: Dict[str, Any]) -> None:
        self.base_model_creator = base_model_creator
        self.lora_config = lora_config
    
    def __call__(self) -> nn.Module:
        config = LoraConfig(**self.lora_config)
        model = self.base_model_creator()
        lora_model = get_peft_model(model, config)
        return lora_model
