from abc import ABC, abstractmethod
from typing import Dict, List, Union

import torch

from ..core.base import Function

class Loss(Function, ABC):
    @abstractmethod
    def __call__(self, *, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        ...