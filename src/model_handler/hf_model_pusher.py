from torch import nn

from .model_handler import ModelHandler

class HFModelPusher(ModelHandler):
    def __init__(self, *, path: str) -> None:
        self.path = path
    
    def __call__(self, *, model: nn.Module) -> None:
        model.push_to_hub(self.path)