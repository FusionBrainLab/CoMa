from typing import Protocol, Dict, List, Generator, Any
from abc import ABC, abstractmethod
import os

import torch

from ..core.base import Function

class DataLoader(Function, ABC):
    @abstractmethod
    def __call__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        ...