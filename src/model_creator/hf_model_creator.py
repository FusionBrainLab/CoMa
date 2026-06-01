from typing import Dict, Any

from torch import nn
from transformers import (
    AutoModelForCausalLM, 
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoModel,
    AutoModelForImageTextToText,
    Qwen3_5ForConditionalGeneration
)

from .model_creator import ModelCreator

class HFModelCreator(ModelCreator):
    def __init__(self, *, model_cls: str, model_args: Dict[str, Any]) -> None:
        self.model_cls = model_cls
        self.model_args = model_args
    
    def __call__(self) -> nn.Module:
        model = globals()[self.model_cls].from_pretrained(**self.model_args)
        return model