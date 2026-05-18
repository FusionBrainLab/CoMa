from typing import Protocol, Dict, List, Generator, Any
from abc import ABC, abstractmethod
import os
import math

import torch

from .tensor_batch_processor import TensorBatchProcessor

class CompositionTensorProcessor(TensorBatchProcessor):
    def __init__(self, *, base_processors: List[TensorBatchProcessor]) -> None:
        self.base_processors = base_processors

    def __call__(self, *, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for processor in self.base_processors:
            batch = processor(batch=batch)
        return batch
