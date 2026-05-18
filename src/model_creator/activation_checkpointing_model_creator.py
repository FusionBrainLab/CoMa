from typing import List, Any, Union
import re

from torch import nn
#from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from fairscale.nn.checkpoint import checkpoint_wrapper

from .model_creator import ModelCreator
from ..core.attr_access import AttrGetter, AttrSetter
from ..torch_nn_module import CheckpointModel

class ActivationCheckpointingModelCreator(ModelCreator):
    def __init__(self, *, base_model_creator: ModelCreator,
                 submodule_patterns: List[str]) -> None:
        self.base_model_creator = base_model_creator
        self.submodule_patterns = submodule_patterns
    
    def __call__(self) -> nn.Module:
        base_model = self.base_model_creator()
        attr_setter = AttrSetter()
        for pattern in self.submodule_patterns:
            for name, module in base_model.named_modules():
                if re.fullmatch(pattern, name) != None:
                    path = name.split(".")
                    wrapped_submodule = checkpoint_wrapper(module, offload_to_cpu=True)
                    attr_setter(base_model, path, wrapped_submodule)
        return base_model