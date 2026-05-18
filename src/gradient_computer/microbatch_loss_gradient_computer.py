from typing import Dict, Any

import torch
from torch import nn

from ..loss import Loss
from .gradient_computer import GradientComputer

class MicrobatchLossGradientComputer(GradientComputer):
    n_accumulation_steps: int
    loss: Loss
    last_loss: float
    def __init__(self, *, loss: Loss, 
                    n_accumulation_steps: int) -> None:
        self.n_accumulation_steps = n_accumulation_steps
        self.loss = loss
        self.last_loss = 0.0
        
    def __call__(self, *, model: nn.Module, batch: Dict[str, torch.Tensor]) -> None:
        keys = list(batch.keys())
        chunks = {k: batch[k].chunk(self.n_accumulation_steps, dim=0) for k in keys}
        n_microbatches = len(chunks[keys[0]])
        microbatches = [{k: chunks[k][i] for k in keys} for i in range(n_microbatches)]
        total_loss = 0.0
        for microbatch in microbatches:
            pred = model(**microbatch)
            device = pred[list(pred.keys())[0]].device
            for key in microbatch.keys():
                microbatch[key] = microbatch[key].to(device)
            state = {}
            state.update(pred)
            state.update(microbatch)
            loss = self.loss(state=state)
            loss = loss / n_microbatches
            total_loss += loss.detach().item()
            loss.backward()
        self.last_loss = total_loss