from typing import List, TypedDict, Dict, Any
import base64
from pathlib import Path
import io

import torch

from .data_collator import DataCollator
from ..memory_tokenizer import MemoryTokenizer

class MemoryTokenizerCollator(DataCollator):
    def __init__(self, *, memories_col: str,
                        memory_tokenizer: MemoryTokenizer) -> None:
        self.memories_col = memories_col
        self.memory_tokenizer = memory_tokenizer

    def __call__(self, *, batch: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        memories = batch[self.memories_col]
        output = self.memory_tokenizer(memories=memories)
        return output
