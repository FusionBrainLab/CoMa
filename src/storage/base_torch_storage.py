from typing import Any, List, Dict
import json
import os

import torch

from .storage import Storage
from ..torch_state_manager import BaseStateManager

class BaseTorchStorage(Storage):
    def __init__(self, *, relative_path: str) -> None:
        self.path = None
        self.relative_path = relative_path
    
    def set_path(self, *, path: str) -> None:
        self.path = path
    
    def load(self) -> object:
        assert self.path != None
        full_path = os.path.join(self.path, self.relative_path)
        state_manager = BaseStateManager(path=full_path)
        return state_manager

    def save(self, *, obj: object) -> None:
        assert isinstance(obj, dict)
        assert self.path != None
        full_path = os.path.join(self.path, self.relative_path)
        state_manager = BaseStateManager(path=full_path)
        state_dict = state_manager.get_state(statefuls=obj)
        torch.save(state_dict[list(state_dict.keys())[0]], full_path)