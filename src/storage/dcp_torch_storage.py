from typing import Any, List, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter
)

from .storage import Storage
from ..torch_state_manager import DCPTorchStateManager

class DCPTorchStorage(Storage):
    def __init__(self, *, relative_path: str,
                 state_manager_cls: type,
                 full_state_dict: bool,
                 cpu_offload: bool,
                 strict: bool) -> None:
        self.path = None
        self.relative_path = relative_path
        assert issubclass(state_manager_cls, DCPTorchStateManager)
        self.state_manager_cls = state_manager_cls
        self.full_state_dict = full_state_dict
        self.cpu_offload = cpu_offload
        self.strict = strict
        
    def set_path(self, *, path: str) -> None:
        self.path = path
    
    def load(self) -> object:
        assert self.path != None
        full_path = os.path.join(self.path, self.relative_path)
        reader = FileSystemReader(full_path)
        state_manager = self.state_manager_cls(
            reader=reader, full_state_dict=self.full_state_dict, cpu_offload=self.cpu_offload, strict=self.strict
        )
        return state_manager
    
    def save(self, *, obj: object) -> None:
        assert isinstance(obj, dict), f"Object of type {type(obj)} must be dict. "
        assert self.path != None
        full_path = os.path.join(self.path, self.relative_path)
        reader = FileSystemReader(full_path)
        state_manager = self.state_manager_cls(
            reader=reader, full_state_dict=self.full_state_dict, cpu_offload=self.cpu_offload, strict=self.strict
        )
        state_dict = state_manager.get_state(statefuls=obj)
        writer = FileSystemWriter(full_path)
        dcp.save(state_dict, storage_writer=writer)