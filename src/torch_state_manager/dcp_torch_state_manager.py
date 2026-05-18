from abc import ABC, abstractmethod
from typing import Any, Protocol, Dict

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    StorageReader,
    StorageWriter
)

from .torch_state_manager import TorchStateManager, StatefulProtocol

class DCPTorchStateManager(TorchStateManager, ABC):
    def __init__(self, *, reader: StorageReader,
                 full_state_dict: bool,
                 cpu_offload: bool,
                 strict: bool) -> None:
        self.reader = reader
        self.full_state_dict = full_state_dict
        self.cpu_offload = cpu_offload
        self.strict = strict