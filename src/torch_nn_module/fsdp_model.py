from typing import Any, List, Union
import os
from dataclasses import dataclass
import re

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import OffloadPolicy, CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from ..fsdp_model_schedule import FSDPModelSchedule
from ..core.attr_access import AttrGetter
from ..torch_state_manager import DCPModelStateManager
from ..model_creator import ModelCreator

@dataclass
class FSDPModuleShardingConfig:
    submodule_patterns: List[str]
    gradient_sync_schedule: FSDPModelSchedule
    post_backward_reshard_schedule: FSDPModelSchedule
    reshard_after_forward: bool

class FSDPModel(nn.Module):
    base_model: nn.Module
    forward_counter: int
    def __init__(self, *, base_model_creator: ModelCreator,
                 submodule_sharding_configs: List[FSDPModuleShardingConfig],
                 mixed_precision_policy: MixedPrecisionPolicy,
                 offload_policy: OffloadPolicy,
                 place_inputs_on_device: bool) -> None:
        super().__init__()
        self.base_model = base_model_creator()
        self.submodule_sharding_configs = submodule_sharding_configs
        self.mixed_precision_policy = mixed_precision_policy
        self.offload_policy = offload_policy
        self.place_inputs_on_device = place_inputs_on_device
        self.forward_counter = 0

        self.attr_getter = AttrGetter()
    
    def forward(self, **kwargs: Any) -> Any:
        for module_sharding_config in self.submodule_sharding_configs:
            submodule_patterns = module_sharding_config.submodule_patterns
            for pattern in submodule_patterns:
                if pattern == "":
                    self.base_model.set_requires_gradient_sync(module_sharding_config.gradient_sync_schedule(forward=self.forward_counter))
                    self.base_model.set_reshard_after_backward(module_sharding_config.post_backward_reshard_schedule(forward=self.forward_counter))
                    continue
                for name, module in self.base_model.named_modules():
                    if re.fullmatch(pattern, name) != None:
                        module.set_requires_gradient_sync(module_sharding_config.gradient_sync_schedule(forward=self.forward_counter))
                        module.set_reshard_after_backward(module_sharding_config.post_backward_reshard_schedule(forward=self.forward_counter))
        self.forward_counter += 1
        if self.place_inputs_on_device:
            for key in kwargs.keys():
                kwargs[key] = kwargs[key].to(self.device)
        return self.base_model(**kwargs)
    
    def parallelize(self, *, device_mesh: DeviceMesh) -> None:
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.base_model.to(self.device)
        for module_sharding_config in self.submodule_sharding_configs:
            submodule_patterns = module_sharding_config.submodule_patterns
            for pattern in submodule_patterns:
                if pattern == "":
                    fsdp_config = {
                            "mesh":device_mesh,
                            "mp_policy":self.mixed_precision_policy,
                            "offload_policy":self.offload_policy,
                            "reshard_after_forward":module_sharding_config.reshard_after_forward
                        }
                    fully_shard(self.base_model, **fsdp_config)
                    continue
                for name, module in self.base_model.named_modules():
                    if re.fullmatch(pattern, name) != None:
                        fsdp_config = {
                            "mesh":device_mesh,
                            "mp_policy":self.mixed_precision_policy,
                            "offload_policy":self.offload_policy,
                            "reshard_after_forward":module_sharding_config.reshard_after_forward
                        }
                        fully_shard(module, **fsdp_config)

        """device = "cpu" if isinstance(self.offload_policy, CPUOffloadPolicy) else self.device
        self.base_model.to(device=device)"""
        """for name, param in self.base_model.named_parameters():
            if not isinstance(param, torch.distributed.tensor.DTensor):
                print(self.local_rank, name)
        raise"""
        
class CheckpointableFSDPModel(FSDPModel):
    def load_checkpoint(self, *, checkpoint: object) -> None:
        assert isinstance(checkpoint, DCPModelStateManager)
        checkpoint.set_state(statefuls={"model":self.base_model})
    
    def save_checkpoint(self) -> object:
        return {"model":self.base_model}   

