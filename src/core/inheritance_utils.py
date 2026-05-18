from typing import Optional, Callable, TypeVar, ParamSpec, Any, List, Dict, get_type_hints
from abc import ABCMeta, ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
import inspect
from copy import copy

from .object_utils import TotalClassNamespaceParser

class Decorator(ABC):
    @abstractmethod
    def __call__(self, obj: Any) -> Any:
        ...

class Descriptor():
    owner: type
    attribute_name: str
    private_name: str
    decorator: Decorator
    def __init__(self, decorator: Decorator):
        self.decorator = decorator
    
    def __set_name__(self, owner: type, name: str) -> None:
        self.owner = owner
        self.attribute_name = name
        self.private_name = Descriptor.get_private_name(name=name)

    def __get__(self, obj: object, objtype: type = None) -> Any:
        return getattr(obj, self.private_name, None)

    def __set__(self, obj: object, value: Any) -> None:
        setattr(obj, self.private_name, self.decorator(value))
    
    @classmethod
    def get_private_name(cls, name):
        return f"_{name}"

class ChainDecorator(Decorator):
    decorators: List[Decorator]
    def __init__(self, decorators: List[Decorator]):
        self.decorators = decorators
    
    def __call__(self, obj: Any) -> Any:
        value = obj
        for decorator in self.decorators:
            value = decorator(value)
        return value

@dataclass
class PreMethodConfig:
    method_name: str
    update_input: bool
    
@dataclass
class PostMethodConfig:
    method_name: str
    use_output: bool
    update_output: bool

class PrePostDecorator(Decorator):
    pre_method_config: Optional[PreMethodConfig]
    post_method_config: Optional[PostMethodConfig]
    def __init__(self, *, pre_method_config: Optional[PreMethodConfig], 
                 post_method_config: Optional[PostMethodConfig]) -> None:
        self.pre_method_config = pre_method_config
        self.post_method_config = post_method_config
        
    def __call__(self, obj: Any) -> Any:
        P = ParamSpec("P")
        R = TypeVar("R")
        @wraps(obj)
        def decorated_func(*args: P.args, **kwargs: P.kwargs) -> R:
            if self.pre_method_config:
                pre_method = getattr(args[0], self.pre_method_config.method_name)
                pre_output = pre_method(**kwargs)
                if self.pre_method_config.update_input:
                    kwargs.update(pre_output)
            output = obj(*args, **kwargs)
            if self.post_method_config:
                post_method = getattr(args[0], self.post_method_config.method_name)
                if self.post_method_config.use_output:
                    kwargs.update({"OUTPUT": output})
                post_output = post_method(**kwargs)
                if self.post_method_config.update_output:
                    output.update(post_output)
            return output
        return decorated_func

@dataclass
class MethodDecorationConfig:
    method_name: str
    method_decorator: Decorator

@dataclass
class AttributeDescriptionConfig:
    attribute_name: str
    attribute_descriptor: Descriptor

class DynamicSubclassDecorator(Decorator):
    metaclass: type
    method_decoration_configs: List[MethodDecorationConfig]
    attribute_description_configs: List[AttributeDescriptionConfig]
    subclass_prefix: str
    def __init__(self, metaclass: type,
                    method_decoration_configs: List[MethodDecorationConfig],
                    attribute_description_configs: List[AttributeDescriptionConfig],
                    subclass_prefix: str) -> None:
        self.metaclass = metaclass
        self.method_decoration_configs = method_decoration_configs
        self.attribute_description_configs = attribute_description_configs
        self.subclass_prefix = subclass_prefix
    
    def __call__(self, obj: Any) -> Any:
        objtype = type(obj)
        objtype_name = type(obj).__name__
        namespace = {}
        attributes = []
        for config in self.method_decoration_configs:
            namespace[config.method_name] = config.method_decorator
        for config in self.attribute_description_configs:
            namespace[config.attribute_name] = config.attribute_descriptor
            attributes.append(config.attribute_name)
        namespace["__annotations__"] = {k: v for k, v in get_type_hints(objtype).items() if k in attributes}
        child_class = self.metaclass(
            f"{self.subclass_prefix}{objtype_name}",
            (objtype,),
            namespace
        )
        new_obj = copy(obj)
        new_obj.__class__ = child_class
        return new_obj

