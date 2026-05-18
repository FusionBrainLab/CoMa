from abc import ABC, abstractmethod
from typing import Any, Protocol, Dict

import torch

from .torch_state_manager import TorchStateManager, StatefulProtocol

class BaseStateManager(TorchStateManager):
    def __init__(self, *, path: str) -> None:
        self.path = path
        
    def set_state(self, *, statefuls: Dict[str, StatefulProtocol]) -> None:
        assert len(statefuls.keys()) == 1
        state_dict = torch.load(self.path)
        statefuls[list(statefuls.keys())[0]].load_state_dict(state_dict)
    
    def get_state(self, *, statefuls: Dict[str, StatefulProtocol]) -> Dict[str, Any]:
        assert len(statefuls.keys()) == 1
        state_dict = {}
        for key, value in statefuls.items():
            state_dict[key] = value.state_dict()
        return state_dict