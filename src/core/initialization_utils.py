from abc import ABC, abstractmethod
import inspect
from typing import List, Dict, Any, Tuple, Set

from typeguard import typechecked

from .inheritance_utils import Decorator, Descriptor, ChainDecorator
from .object_utils import TotalClassNamespaceParser, DirectClassDictNamespaceParser, InheritedClassNamespaceParser, DirectClassNamespaceParser, BaseClassNamespaceParser

class Initialization(ABC):
    """Base class for all initialization logics. Any subclass call will return new namespace based on class current namespace and it's bases. """
    @abstractmethod
    def __call__(self, name: str, bases: List[type], namespace: Dict[str, Any]) -> Dict[str, Any]:
        ...

class InheritanceInitialization(Initialization):
    def __call__(self, name: str, bases: List[type], namespace: Dict[str, Any]) -> Dict[str, Any]:
        inherited_parser = InheritedClassNamespaceParser()
        direct_parser = DirectClassNamespaceParser()
        direct_namespace_parser = DirectClassDictNamespaceParser()
        
        direct_namespace = direct_namespace_parser(namespace)
        base_namespaces = [inherited_parser(base) + direct_parser(base) for base in bases]
        subclass_namespace = {"__annotations__": namespace.get("__annotations__", {})}
        for config in direct_namespace:
            var_name = config.name
            if var_name in ["__abstractmethods__"]:
                continue
            value = config.value
            base_value = None
            for base_namespace in base_namespaces:
                base_names = {c.name: c for c in base_namespace}
                if var_name in base_names:
                    base_value = base_names[var_name].value
                    break
            if base_value == None:
                if value is Decorator:
                    raise TypeError(
                        f"Base class doesn't have method {var_name} to decorate. "
                    )
                subclass_namespace[var_name] = value
            else:
                if (inspect.ismethod(base_value) or inspect.isfunction(base_value)):
                    safe_methods = ["__init__"]
                    if issubclass(type(value), Decorator):
                        subclass_namespace[var_name] = value(base_value)
                    elif getattr(base_value, "__isabstractmethod__", False) and (inspect.ismethod(value) or inspect.isfunction(value)) or var_name in safe_methods:
                        subclass_namespace[var_name] = value
                    else:
                        raise TypeError(
                            f"Can't override attribute '{var_name}' with value: {value}"
                        )
                elif ((type(value) == Descriptor or issubclass(type(value), Descriptor)) and 
                      (type(base_value) == Descriptor or issubclass(type(value), Descriptor))):
                    new_decorator = ChainDecorator([base_value.decorator, value.decorator])
                    value.decorator = new_decorator
                    subclass_namespace[var_name] = value
                else:
                    raise TypeError(
                        f"Can't override attribute '{var_name}' with value: {value}"
                    )
        return subclass_namespace

class TypecheckingInitialization(Initialization):
    def __call__(self, name: str, bases: List[type], namespace: Dict[str, Any]) -> Dict[str, Any]:
        new_namespace = {k: v for k, v in namespace.items()}
        for member_name, member in namespace.items():
            if inspect.isfunction(member) or inspect.ismethod(member):
                if not getattr(member, "__isabstractmethod__", False):
                    new_namespace[member_name] = typechecked(member)
        """parent_method = None
        total_namespace_parser = TotalClassNamespaceParser()
        for base in bases:
            base_namespace = total_namespace_parser(base)
            base_dict = {v.name: v for v in base_namespace}
            if "__setattr__" in base_dict:
                parent_method = base_dict["__setattr__"].value
                break
        assert parent_method != None
        new_namespace["__setattr__"] = typechecked(parent_method)"""
        return new_namespace