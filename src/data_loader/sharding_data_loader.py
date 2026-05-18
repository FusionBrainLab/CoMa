from typing import Protocol, Dict, List, Generator, Any
import os

import torch
from torch.distributed.device_mesh import DeviceMesh

from .data_loader import DataLoader
from ..batch_sharder import BatchSharder

class ShardingDataLoader(DataLoader):
    base_dataloader: DataLoader
    def __init__(self, *, base_dataloader: DataLoader,
                batch_sharding_policy: BatchSharder,
                place_on_device: bool) -> None:
        self.base_dataloader = base_dataloader
        self.batch_sharding_policy = batch_sharding_policy
        self.place_on_device = place_on_device
        self.device_mesh = None
        
        self.samples_passed = 0
        self.batches_passed = 0
    
    def __call__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        assert self.device_mesh != None, "ShardingDataLoader must be parallelized before calling. "
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.device = torch.device(f"cuda:{self.local_rank}")
        for batch in self.base_dataloader():
            keys = list(batch.keys())
            batch_len = len(batch[keys[0]])
            num_shards = self.device_mesh.size()
            batch_shard = self.batch_sharding_policy(batch=batch, n_shards=num_shards)[self.local_rank]
            self.batches_passed += 1
            self.samples_passed += batch_len
            yield batch_shard
    
    def parallelize(self, *, device_mesh: DeviceMesh) -> None:
        self.device_mesh = device_mesh
    
class CheckpointableShardingDataLoader(ShardingDataLoader):
    def load_checkpoint(self, *, checkpoint: object) -> None:
        self.base_dataloader.load_checkpoint(checkpoint=checkpoint)
    
    def save_checkpoint(self) -> object:
        return self.base_dataloader.save_checkpoint()