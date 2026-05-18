from abc import ABC, abstractmethod
from typing import Any, Protocol, Dict

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
)

from .dcp_torch_state_manager import DCPTorchStateManager
from .torch_state_manager import StatefulProtocol

class DCPOptimizerStateManager(DCPTorchStateManager):
    def set_state(self, *, statefuls: Dict[str, StatefulProtocol]) -> None:
        assert "model" in statefuls and "optimizer" in statefuls
        options = StateDictOptions(full_state_dict=self.full_state_dict, cpu_offload=self.cpu_offload)
        state_dict = get_optimizer_state_dict(statefuls["model"], statefuls["optimizer"], options=options)
        dcp.load(state_dict=state_dict, storage_reader=self.reader)
        
    def get_state(self, *, statefuls: Dict[str, StatefulProtocol]) -> Dict[str, Any]:
        assert "model" in statefuls and "optimizer" in statefuls
        options = StateDictOptions(full_state_dict=self.full_state_dict, cpu_offload=self.cpu_offload)
        return get_optimizer_state_dict(statefuls["model"], statefuls["optimizer"], options=options)