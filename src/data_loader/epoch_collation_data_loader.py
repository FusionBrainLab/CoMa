from typing import Protocol, Dict, List, Generator, Any
from abc import ABC, abstractmethod
import os
import math

import torch

from .data_loader import DataLoader
from ..dataset_creator import DatasetCreator

class DataCollator(Protocol):
    def __call__(self, *, batch: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        pass

class EpochCollationDataLoader(DataLoader):
    dataset: Dict[str, List[Any]]
    n_epochs: int
    batch_size: int
    collator: DataCollator
    
    length: int
    samples_passed: int
    batches_passed: int
    epochs_passed: int
    cur_epoch: int
    def __init__(self, *, dataset_creator: DatasetCreator, 
                 n_epochs: int, 
                 batch_size: int, 
                 collator: DataCollator,
                 drop_last_batch: bool) -> None:
        self.dataset = dataset_creator()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.collator = collator
        self.drop_last_batch = drop_last_batch
        
        self.dataset_length = len(self.dataset[list(self.dataset.keys())[0]])
        self.length = math.ceil(self.dataset_length / self.batch_size) * self.n_epochs if not self.drop_last_batch else math.floor(self.dataset_length / self.batch_size) * self.n_epochs
        self.samples_passed = 0
        self.batches_passed = 0
        self.epochs_passed = 0
        self.cur_epoch = 0
    
    def __call__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        for epoch in range(self.n_epochs):
            self.cur_epoch += 1
            n_batches = int(self.length / self.n_epochs)
            batch_inds = [[j for j in range(i * self.batch_size, min((i + 1) * self.batch_size, self.dataset_length))] for i in range(n_batches)]
            batches = [{k: [self.dataset[k][ind] for ind in binds] for k in self.dataset.keys()} for binds in batch_inds]
            for batch in batches:
                batch = self.collator(batch=batch)
                self.batches_passed += 1
                self.samples_passed += len(batch[list(batch.keys())[0]])
                yield batch
            self.epochs_passed += 1