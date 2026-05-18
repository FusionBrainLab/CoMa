from abc import ABC, abstractmethod
from typing import Dict, List, Union

import torch
from torch import nn

from ..loss import Loss
from ..core.attr_access import AttrGetter

class LMLoss(Loss):
    logits_path: List[Union[str, int]]
    labels_path: List[Union[str, int]]
    cross_entropy: nn.CrossEntropyLoss
    def __init__(self, *, logits_path: List[Union[str, int]], labels_path: List[Union[str, int]]) -> None:
        self.logits_path = logits_path
        self.labels_path = labels_path
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def __call__(self, *, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        attr_getter = AttrGetter()
        labels = attr_getter(state, self.labels_path)[:, 1:].contiguous()
        logits = attr_getter(state, self.logits_path).contiguous()
        logits = logits[..., :-1, :]
        logits = logits.permute(0, 2, 1)
        loss = self.cross_entropy(logits, labels)
        return loss