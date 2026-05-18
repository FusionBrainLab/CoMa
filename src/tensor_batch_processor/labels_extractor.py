from typing import Protocol, Dict, List, Generator, Any
from abc import ABC, abstractmethod
import os
import math

import torch

from .tensor_batch_processor import TensorBatchProcessor

class LabelsExtractor(TensorBatchProcessor):
    def __init__(self, *, input_ids_key: str,
                        start_token_id: int,
                        end_token_id: int,
                        ignore_index: int) -> None:
        self.input_ids_key = input_ids_key
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.ignore_index = ignore_index

    def __call__(self, *, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch[self.input_ids_key]

        labels = torch.full_like(input_ids, self.ignore_index)

        input_ids_flat = input_ids[0].tolist()
        L = len(input_ids_flat)
        pos = 0
        while pos < L:
            if input_ids_flat[pos] == self.start_token_id:
                ans_start = pos + 2
                ans_end = ans_start
                while ans_end < L and input_ids_flat[ans_end] != self.end_token_id:
                    ans_end += 1
                if ans_end < L:
                    labels[0, ans_start : ans_end + 2] = input_ids[
                        0, ans_start : ans_end + 2
                    ]
                    pos = ans_end
            pos += 1

        batch["labels"] = labels
        return batch