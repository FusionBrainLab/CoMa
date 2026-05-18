from typing import List, TypedDict, Dict, Any
import base64
from pathlib import Path
import io

import torch

from .data_collator import DataCollator
from ..tensor_batch_processor import TensorBatchProcessor

class PostprocessCollator(DataCollator):
    def __init__(self, *, base_collator: DataCollator,
                        postprocessor: TensorBatchProcessor) -> None:
        self.base_collator = base_collator
        self.postprocessor = postprocessor

    def __call__(self, *, batch: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        tensor_batch = self.base_collator(batch=batch)
        output = self.postprocessor(batch=tensor_batch)
        return output