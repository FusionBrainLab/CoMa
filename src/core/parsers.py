from typing import Any, List, Dict, Union
from abc import ABC, abstractmethod

from .base import Function
from .attr_access import AttrGetter

class Parser(Function, ABC):
    @abstractmethod
    def __call__(self, *, inputs: Any) -> Any:
        ...

class AnyToDictParser(Parser):
    def __init__(self, *, key: str) -> None:
        self.key = key
    
    def __call__(self, *, inputs: Any) -> Dict[str, Any]:
        return {self.key: inputs}

class DictToDictParser(Parser):
    def __init__(self, *, keys_mapping: Dict[str, str]) -> None:
        self.keys_mapping = keys_mapping
    
    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {k: inputs[self.keys_mapping[k]] for k in self.keys_mapping.keys()}

class AnyToDictPathParser(Parser):
    def __init__(self, *, key_paths: Dict[str, List[Union[str,int]]]) -> None:
        self.key_paths = key_paths
        self.attr_getter = AttrGetter()
    
    def __call__(self, *, inputs: Any) -> Dict[str, Any]:
        return {k: self.attr_getter(inputs, v) for k, v in self.key_paths.items()}

class AnyToListParser(Parser):
    def __call__(self, *, inputs: Any) -> List[Any]:
        return [inputs]

class AnyToListPathParser(Parser):
    def __init__(self, *, item_paths: List[List[Union[str,int]]]) -> None:
        self.item_paths = item_paths
        self.attr_getter = AttrGetter()
    
    def __call__(self, *, inputs: Any) -> List[Any]:
        return [self.attr_getter(inputs, path) for path in self.item_paths]

class DictToListParser(Parser):
    def __init__(self, *, keys_order: List[str]) -> None:
        self.keys_order = keys_order
        
    def __call__(self, *, inputs: Dict[str, Any]) -> List[Any]:
        output = [inputs[k] for k in self.keys_order]
        return output

class DictToAnyParser(Parser):
    def __init__(self, *, key: str) -> None:
        self.key = key
    
    def __call__(self, *, inputs: Dict[str, Any]) -> Any:
        return inputs[self.key]