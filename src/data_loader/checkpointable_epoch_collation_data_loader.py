from typing import Protocol, Dict, List, Generator, Any
import os

import torch

from .epoch_collation_data_loader import EpochCollationDataLoader
from ..core.inheritance_utils import PrePostDecorator, PreMethodConfig, PostMethodConfig

class CheckpointableEpochCollationDataLoader(EpochCollationDataLoader):
    __call__ = PrePostDecorator(pre_method_config=None, post_method_config=PostMethodConfig(method_name="postcall", use_output=True, update_output=False))
    
    def postcall(self, *, OUTPUT: Generator[Dict[str, torch.Tensor], None, None]) -> None:
        if getattr(self, "checkpoint", None) != None:
            i = 1
            for _ in OUTPUT:
                if i == self.checkpoint["batches_passed"]:
                    break
                i += 1
    
    def load_checkpoint(self, *, checkpoint: object) -> None:
        self.checkpoint = checkpoint
    
    def save_checkpoint(self) -> object:
        checkpoint = {"batches_passed":self.batches_passed}
        return checkpoint