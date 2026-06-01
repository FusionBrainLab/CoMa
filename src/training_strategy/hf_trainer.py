import os
from typing import Optional, TypedDict, Dict, Any, List
from functools import partial

from torch import nn
from transformers import Trainer
import torch
from transformers.training_args import TrainingArguments 

from .training_strategy import TrainingStrategy
from ..dataset_creator import DatasetCreator
from ..torch_dataset import DictDataset
from ..data_collator import DataCollator
from ..hf_processor_creator import HFProcessorCreator

class HFTrainer(TrainingStrategy):
    def __init__(self, *, init_args: Dict[str, Any],
                        enable_input_require_grads: bool,
                        processor_creator: HFProcessorCreator| None,
                        use_cache: bool,
                        train_dataset_creator: DatasetCreator,
                        data_collator: DataCollator,
                        resume_from_checkpoint: bool) -> None:
        self.init_args = init_args
        self.enable_input_require_grads = enable_input_require_grads
        self.processor_creator = processor_creator
        self.use_cache = use_cache
        self.train_dataset_creator = train_dataset_creator
        self.data_collator = data_collator
        self.resume_from_checkpoint = resume_from_checkpoint

    def __call__(self, *, model: nn.Module) -> nn.Module:
        if self.enable_input_require_grads:
            model.enable_input_require_grads()
        #model.to(f"cuda:{torch.cuda.current_device()}")
        #model.model.language_model.embed_tokens.to(f"cuda:{torch.cuda.current_device()}")

        model.config.use_cache = self.use_cache
        #model = model.cuda()

        train_dataset = self.train_dataset_creator()
        train_dataset = DictDataset(dict_dataset=train_dataset)
        
        processing_class = self.processor_creator() if self.processor_creator != None else None
        # transformers>=5.5: TensorBoardCallback reads TENSORBOARD_LOGGING_DIR, not TrainingArguments.logging_dir
        logging_dir = self.init_args.get("logging_dir")
        if logging_dir is not None:
            os.environ["TENSORBOARD_LOGGING_DIR"] = os.path.expanduser(logging_dir)
        args = TrainingArguments(**self.init_args)

        def collate_fn(instances: List[Dict[str, Any]]) -> Dict[str, Any]:
            keys = list(instances[0].keys())
            batch = {k: [instances[i][k] for i in range(len(instances))] for k in keys}
            results = self.data_collator(batch=batch)
            """for k in results.keys():
                results[k] = results[k].to(f"cuda:{torch.cuda.current_device()}")""" #if type(results[k]) != torch.cuda.LongTensor else results[k]
            return results

        trainer = Trainer(
            model=model,
            processing_class=processing_class,
            args=args,
            train_dataset=train_dataset,
            data_collator=collate_fn
        )
        trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)

        trainer.save_state()
        return trainer.model