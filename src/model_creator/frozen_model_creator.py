from typing import List, Any, Union
import re

from torch import nn

from .model_creator import ModelCreator
from ..core.attr_access import AttrGetter, AttrSetter

class FrozenModelCreator(ModelCreator):
    def __init__(self, *, base_model_creator: ModelCreator,
                 frozen_submodule_patterns: List[str]) -> None:
        self.base_model_creator = base_model_creator
        self.frozen_submodule_patterns = frozen_submodule_patterns
    
    def __call__(self) -> nn.Module:
        base_model = self.base_model_creator()
        attr_setter = AttrSetter()
        for pattern in self.frozen_submodule_patterns:
            for name, module in base_model.named_modules():
                if re.fullmatch(pattern, name) != None:
                    for n, p in module.named_parameters():
                        p.requires_grad = False
        return base_model