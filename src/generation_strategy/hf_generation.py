from typing import List, Any, Dict

import pandas as pd
import torch

from .generation_strategy import GenerationStrategy
from ..model_creator import ModelCreator
from ..hf_processor_creator import HFProcessorCreator

class HFGeneration(GenerationStrategy):
    def __init__(self, *, model_creator: ModelCreator, 
                        tokenizer_creator: HFProcessorCreator,
                        generation_args: Dict[str, Any],
                        device: str,
                        output_key: str) -> None:
        self.model = model_creator()
        self.tokenizer = tokenizer_creator()
        self.generation_args = generation_args
        self.device = device
        self.output_key = output_key

        self.model.to(self.device)
        
    def __call__(self, *, sample: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in list(sample.items()):
            if isinstance(v, torch.Tensor):
                sample[k] = v.to(self.device)

        generated = self.model.generate(**sample, **self.generation_args)[:, sample["input_ids"].shape[1]:]
        outputs = []
        for i, out_ids in enumerate(generated):
            decoded = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            outputs.append(decoded)
        output = {self.output_key: outputs}
        return output