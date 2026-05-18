from torch import nn

from .model_handler import ModelHandler

class HFModelSaver(ModelHandler):
    def __init__(self, *, path: str) -> None:
        self.path = path
    
    def __call__(self, *, model: nn.Module) -> None:
        model.save_pretrained(self.path)