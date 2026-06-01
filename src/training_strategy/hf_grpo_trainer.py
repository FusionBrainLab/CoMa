from typing import Optional, TypedDict, Dict, Any, List
from functools import partial, wraps, update_wrapper

from torch import nn
from transformers import Trainer
import torch
from transformers.training_args import TrainingArguments 

from .training_strategy import TrainingStrategy
from ..dataset_creator import DatasetCreator
from ..torch_dataset import DictDataset
from ..data_collator import DataCollator
from ..hf_processor_creator import HFProcessorCreator
from ..core.base import Function

class HFGRPOTrainer(TrainingStrategy):
    def __init__(self, *, init_args: Dict[str, Any],
                        enable_input_require_grads: bool,
                        processor_creator: HFProcessorCreator| None,
                        use_cache: bool,
                        train_dataset_creator: DatasetCreator,
                        resume_from_checkpoint: bool,
                        reward_functions: Dict[str, Function]) -> None:
        self.init_args = init_args
        self.enable_input_require_grads = enable_input_require_grads
        self.processor_creator = processor_creator
        self.use_cache = use_cache
        self.train_dataset_creator = train_dataset_creator
        self.resume_from_checkpoint = resume_from_checkpoint
        self.reward_functions = reward_functions

    def __call__(self, *, model: nn.Module) -> nn.Module:
        from trl import GRPOTrainer, GRPOConfig

        if self.enable_input_require_grads:
            model.enable_input_require_grads()

        model.config.use_cache = self.use_cache

        train_dataset = self.train_dataset_creator()
        train_dataset = DictDataset(dict_dataset=train_dataset)
        #print(train_dataset[0])
        
        processing_class = self.processor_creator() if self.processor_creator != None else None
        args = GRPOConfig(**self.init_args)

        rewards = []
        for key, metric in self.reward_functions.items():
            def create_wrapper(m, name):
                @wraps(m)
                def reward_wrapper(*args, **kwargs):
                    """print(args)
                    print(kwargs)"""
                    return m(**kwargs)
                reward_wrapper.__name__ = name
                reward_wrapper.__qualname__ = name
                return reward_wrapper
            rewards.append(create_wrapper(metric, key))

        trainer = GRPOTrainer(
            model=model,
            processing_class=processing_class,
            args=args,
            train_dataset=train_dataset,
            reward_funcs=rewards
        )
        trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)

        trainer.save_state()
        return trainer.model