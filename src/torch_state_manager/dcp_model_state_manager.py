from abc import ABC, abstractmethod
from typing import Any, Protocol, Dict

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from .dcp_torch_state_manager import DCPTorchStateManager
from .torch_state_manager import StatefulProtocol

class DCPModelStateManager(DCPTorchStateManager):
    def set_state(self, *, statefuls: Dict[str, StatefulProtocol]) -> None:
        assert "model" in statefuls
        options = StateDictOptions(full_state_dict=self.full_state_dict, cpu_offload=self.cpu_offload)
        state_dict = get_model_state_dict(statefuls["model"], options=options)
        dcp.load(state_dict=state_dict, storage_reader=self.reader)
        
    def get_state(self, *, statefuls: Dict[str, StatefulProtocol]) -> Dict[str, Any]:
        assert "model" in statefuls
        options = StateDictOptions(full_state_dict=self.full_state_dict, cpu_offload=self.cpu_offload)
        return get_model_state_dict(statefuls["model"], options=options)