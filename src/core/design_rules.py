from abc import ABC, abstractmethod
import inspect
from typing import List, Dict, Any, Tuple, Set, get_origin, get_args, Union, Generator

import numpy as np

from .inheritance_utils import Decorator, Descriptor
from .object_utils import InheritedClassNamespaceParser, DirectClassNamespaceParser, BaseClassNamespaceParser, DirectClassDictNamespaceParser, TotalClassNamespaceParser

BASE_CLASSES = ["Function", "Container", "Parser", "AnyToDictParser", "DictToDictParser", "AnyToDictPathParser", "AnyToListParser", "DictToListParser", 
                "DictToAnyParser", "AnyToListPathParser", "FunctionWrapper", "FunctionGraph", "FunctionMap", "FunctionLoop"]

class Validation(ABC):
    """Base class for all design rules. Any subclass call will check type data and raise TypeError if type is invalid. """
    @abstractmethod
    def __call__(self, name: str, bases: List[type], namespace: Dict[str, Any]) -> None:
        ...

class ArgumentKindsValidation(Validation):
    """Arguments to class methods may either be all KEYWORD_ONLY or must be represented by a single VAR_KEYWORD. """
    def __call__(self, name: str, bases: List[type], namespace: Dict[str, Any]) -> None:
        safe_methods = ["__getattr__", "__setattr__"]
        for name, member in namespace.items():
            if inspect.ismethod(member) or inspect.isfunction(member):
                if name in safe_methods:
                    continue
                sig = inspect.signature(member)
                params = {k: v for k, v in sig.parameters.items() if k != "self" and k != "cls"}
                var_kwargs_count = 0
                
                for param in params.values():
                    if param.kind == inspect.Parameter.VAR_KEYWORD:
                        var_kwargs_count += 1
                    elif param.kind != inspect.Parameter.KEYWORD_ONLY:
                        raise TypeError(
                            f"""Method '{name}' must only use KEYWORD_ONLY or single VAR_KEYWORD arguments, 
                            except only one parameter for 'self' argument, but found '{param.name}' of kind {param.kind}. """
                        )
                
                if var_kwargs_count > 1:
                    raise TypeError(
                        f"Method '{name}' can have at most one VAR_KEYWORD parameter. "
                    )

class TypingExistanceValidation(Validation):
    """All class attributes must have typing, all methods must have argument typing. """
    def __call__(self, name: str, bases: List[type], namespace: Dict[str, Any]) -> None:
        namespace_parser = DirectClassDictNamespaceParser()
        direct_namespace = namespace_parser(namespace)
        for config in direct_namespace:
            var_name = config.name
            member = config.value
            if var_name in ["__classcell__", "__abstractmethods__", "__slotnames__"]:
                continue
            if issubclass(type(member), Decorator):
                continue
            if inspect.ismethod(member) or inspect.isfunction(member):
                sig = inspect.signature(member)
                params = {k: v for k, v in sig.parameters.items() if k != "self" and k != "cls"}
                for param_name, param in params.items():
                    if param.annotation == inspect.Parameter.empty:
                        raise TypeError(
                            f"Method '{var_name}' doesn't have annotation for parameter '{param_name}'. "
                        )
                if sig.return_annotation == inspect.Parameter.empty:
                    raise TypeError(
                        f"Method '{var_name}' doesn't contain return annotation. "
                    )
            else:
                attr_type = config.type
                if attr_type == inspect.Parameter.empty:
                    raise TypeError(
                        f"Attribute '{var_name}' doesn't contain annotation. "
                    )

class DefaultsExistanceValidation(Validation):
    """Classes must not have default values ​​for function arguments. 
    Attributes can have a default value only if the value is a Descriptor or Decorator."""
    def __call__(self, name: str, bases: List[type], namespace: Dict[str, Any]) -> None:
        namespace_parser = DirectClassDictNamespaceParser()
        direct_namespace = namespace_parser(namespace)
        for config in direct_namespace:
            var_name = config.name
            member = config.value
            if var_name in ["__classcell__", "__abstractmethods__", "__slotnames__"]:
                continue
            if inspect.ismethod(member) or inspect.isfunction(member):
                sig = inspect.signature(member)
                for param_name, param in sig.parameters.items():
                    if param.default not in [None, inspect._empty]:
                        raise TypeError(
                            f"Method '{var_name}' must not have arguments with default value, ",
                            f"but found {param_name} with value {param.default}. "
                        )
            else:
                if (not issubclass(type(member), Descriptor) and 
                    type(member) != Descriptor and 
                    not issubclass(type(member), Decorator) and
                    member != None):
                    raise TypeError(
                        f"Attribute '{var_name}' must not have default value"
                    )

class AbstractionValidation(Validation):
    """The class should depend only on interfaces - the types of attributes and method arguments should be abstract or builtin. """
    def __call__(self, name: str, bases: List[type], namespace: Dict[str, Any]) -> None:
        if name in BASE_CLASSES:
            return
        namespace_parser = DirectClassDictNamespaceParser()
        direct_namespace = namespace_parser(namespace)
        for config in direct_namespace:
            var_name = config.name
            var_type = config.type
            member = config.value
            if inspect.ismethod(member) or inspect.isfunction(member):
                sig = inspect.signature(member)
                params = {k: v for k, v in sig.parameters.items() if k != "self" and k != "cls"}
                for param_name, param in params.items():
                    annotation = param.annotation 
                    if not self._is_allowed_type(annotation):
                        raise TypeError(
                            f"Invalid annotation {annotation} for param '{param_name}' of method {var_name}. "
                        )
                return_annotation = sig.return_annotation
                if not self._is_allowed_type(return_annotation):
                    raise TypeError(
                        f"Invalid return annotation {annotation} of method {var_name}. "
                    )
            else:
                if not self._is_allowed_type(var_type):
                    raise TypeError(
                        f"Invalid annotation {annotation} for attribute {var_name}. "
                    )
    
    def _is_allowed_type(self, annotation: type) -> bool:
        valid_typings = [List, Dict, Set, Tuple, Union, Generator]
        is_abstract = inspect.isabstract(annotation)
        is_builtin = getattr(annotation, "__module__", "") == 'builtins'
        is_valid_typing = annotation in valid_typings
        is_none = annotation == None
        is_not_base = getattr(annotation, "__name__", "") not in BASE_CLASSES
        
        origin = get_origin(annotation)
        if origin is not None:
            is_valid_origin = self._is_allowed_type(origin)
            args = get_args(annotation)
            is_valid_args = all(self._is_allowed_type(a) for a in args)
            is_valid_generic = is_valid_origin and is_valid_args
        else:
            is_valid_generic = True
        
        return is_abstract or is_builtin or is_valid_typing or is_none or is_not_base

class AtomizationValidation(Validation):
    """A class must follow either the function pattern or the container pattern. 
    The function pattern implies a single method call and any number of attributes. 
    The container pattern implies no more than one attribute and any number of methods.
    The composition pattern implies a ban on creating new attributes, only overriding the attributes of base classes is allowed."""
    def __call__(self, name: str, bases: List[type], namespace: Dict[str, Any]) -> None:
        pattern_validators = [self._is_valid_function, self._is_valid_container, self._is_valid_composition]
        is_valid = False
        for pattern_validator in pattern_validators:
            if pattern_validator(bases, namespace):
                is_valid = True
                break
        if not is_valid:
            raise TypeError(
                "Class doesn't follow neither function, container or composition pattern. "
            )
    
    def _is_valid_function(self, bases: List[type], namespace: Dict[str, Any]) -> bool:
        methods = []
        for name, member in namespace.items():
            if inspect.ismethod(member) or inspect.isfunction(member):
                methods.append(name)
        return set(methods) in [set(["__call__"]), set(["__init__", "__call__"]), set(["__new__"])]
    
    def _is_valid_container(self, bases: List[type], namespace: Dict[str, Any]) -> bool:
        namespace_parser = DirectClassDictNamespaceParser()
        direct_namespace = namespace_parser(namespace)
        attributes = []
        for config in direct_namespace:
            name = config.name
            member = config.value
            if name in ["__classcell__", "__abstractmethods__"]:
                continue
            if not inspect.ismethod(member) and not inspect.isfunction(member):
                attributes.append(name)
        return len(attributes) <= 1
    
    def _is_valid_composition(self, bases: List[type], namespace: Dict[str, Any]) -> bool:
        base_namespace_parser = BaseClassNamespaceParser()
        inherited_namespace_parser = InheritedClassNamespaceParser()
        direct_namespace_parser = DirectClassNamespaceParser()
        direct_dict_namespace_parser = DirectClassDictNamespaceParser()
        is_valid = True
        for c in direct_dict_namespace_parser(namespace):
            name = c.name
            is_new = True
            for base in bases:
                base_names = [c.name for c in base_namespace_parser(base)]
                if name in base_names:
                    continue
                inherited_names = [c.name for c in inherited_namespace_parser(base)]
                direct_names = [c.name for c in direct_namespace_parser(base)]
                if name in inherited_names or name in direct_names:
                    is_new = False
                    break
            if is_new:
                is_valid = False
                break
        return is_valid
    
class InheritanceValidation(Validation):
    """При мультинаследовании, если один и тот же метод определен в нескольких нодах ориентированного графа наследования, 
    то должен существовать путь из каждой точки определения метода в каждую."""
    def __call__(self, name: str, bases: List[type], namespace: Dict[str, Any]) -> None:
        self._is_valid_inheritance_graph(name, bases, namespace)
        for base in bases:
            self._is_valid_direct_inheritance(base, namespace)

    def _is_valid_inheritance_graph(self, name: str, bases: List[type], namespace: Dict[str, Any]) -> bool:
        graph = self._get_inheritance_graph(name, bases)
        
        dict_parser = DirectClassDictNamespaceParser()
        cls_parser = DirectClassNamespaceParser()
        skip_vars = ["__setattr__", "__abstractmethods__", "_abc_impl"]
        namespaces = {name: [c for c in dict_parser(namespace) if c.name not in skip_vars]}
        for base in bases:
            namespaces[base.__name__] = [c for c in cls_parser(base) if c.name not in skip_vars]
        
        overriddes = {}
        for cls, vars in namespaces.items():
            names = [v.name for v in vars]
            for name in names:
                if name not in overriddes:
                    overriddes[name] = []
                overriddes[name].append(cls)
        
        paths = {}
        for var, classes in overriddes.items():
            local_paths = []
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    local_paths.append([classes[i], classes[j]])
            paths[var] = local_paths
        
        for name, local_paths in paths.items():
            for path in local_paths:
                if not self._is_oriented_graph_path_exists(graph, path[0], path[1]) and not self._is_oriented_graph_path_exists(graph, path[1], path[0]):
                    raise TypeError(
                        f"No path in oriented inheritance graph for var {name}"
                    )
        
    def _get_inheritance_graph(self, name: str, bases: List[type]) -> Dict[str, List[str]]:
        graph = {name: [b.__name__ for b in bases]}
        visited = set()
        visited.add(name)
        
        def _build_graph(current_cls):
            class_name = current_cls.__name__
            if class_name in visited:
                return
            
            visited.add(class_name)
            parents = []
            
            for base in current_cls.__bases__:
                if base.__name__ != 'object':
                    parents.append(base.__name__)
                    _build_graph(base)
            
            graph[class_name] = parents
        
        for base in bases:
            _build_graph(base)
        return graph
        
    def _is_oriented_graph_path_exists(self, graph: np.ndarray, start: str, end: str, visited: Set[str] = None) -> bool:
        """
        Check if a path exists from start to end using DFS (recursive).
        
        Args:
            graph: Dict representing the oriented graph {vertex: [neighbors]}
            start: Starting vertex
            end: Target vertex
            visited: Set of visited vertices (used internally for recursion)
        
        Returns:
            bool: True if path exists, False otherwise
        """
        if visited is None:
            visited = set()
        
        if start == end:
            return True
        
        visited.add(start)
        
        for neighbor in graph.get(start, []):
            if neighbor not in visited:
                if self._is_oriented_graph_path_exists(graph, neighbor, end, visited):
                    return True
                    
        return False
    
    def _is_second_stronger_first(self, first_type, second_type) -> bool:
        """Check if child_type is compatible with parent_type"""
        if first_type is Any:
            return True
        
        # Handle generic types
        parent_origin = get_origin(first_type)
        child_origin = get_origin(second_type)
        
        if parent_origin is not None and child_origin is not None:
            if parent_origin != child_origin:
                return False
            
            parent_args = get_args(first_type)
            child_args = get_args(second_type)
            
            if len(parent_args) != len(child_args):
                return False
                
            return all(self._is_second_stronger_first(p, c) 
                      for p, c in zip(parent_args, child_args))
        
        try:
            return (first_type == second_type or 
                   issubclass(second_type, first_type))
        except TypeError:
            return first_type == second_type
    
    def _is_valid_direct_inheritance(self, base: type, namespace: Dict[str, Any]) -> bool:
        inherited_parser = InheritedClassNamespaceParser()
        direct_parser = DirectClassNamespaceParser()
        direct_dict_parser = DirectClassDictNamespaceParser()
        
        parent_inherited_namespace = inherited_parser(base)
        parent_direct_namespace = direct_parser(base)
        parent_total_namespace = parent_inherited_namespace + parent_direct_namespace
        
        child_direct_namespace = direct_dict_parser(namespace)
        child_vars = {v.name: v for v in child_direct_namespace}
        for var in parent_total_namespace:
            if var.name in child_vars:
                if var.name in ["__abstractmethods__"]:
                    continue
                child_var = child_vars[var.name]
                if inspect.ismethod(var.value) or inspect.isfunction(var.value):
                    if getattr(var.value, "__isabstractmethod__", False):
                        if not inspect.ismethod(child_var.value) and not inspect.isfunction(child_var.value):
                            raise TypeError(
                                f"Abstract method '{var.name}' must be overriden with a function, "
                                f"but have an attribute of '{child_var.type}' in child class. "
                            )
                        else:
                            parent_sig = inspect.signature(var.value)
                            child_sig = inspect.signature(child_var.value)
                            
                            parent_params = {k: v for k, v in parent_sig.parameters.items() if k != "self" and k != "cls"}
                            child_params = {k: v for k, v in child_sig.parameters.items() if k != "self" and k != "cls"}
                            
                            is_single_kwargs_parent = False
                            if len(parent_params) == 1:
                                if list(parent_params.values())[0].kind == inspect.Parameter.VAR_KEYWORD:
                                    is_single_kwargs_parent = True
                            
                            if base.__name__ in BASE_CLASSES and not is_single_kwargs_parent or base.__name__ not in BASE_CLASSES:
                                intersection = set(list(parent_params.keys())).intersection(set(list(child_params.keys())))
                                if len(intersection) != len(parent_params) or len(intersection) != len(child_params):
                                    raise TypeError(f"Different attributes in child and parent classes for '{var.name}' method. ")
                            if base.__name__ not in BASE_CLASSES:
                                for param_name, parent_param in parent_params.items():
                                    child_param = child_params[param_name]  
                                    if parent_param.kind != child_param.kind:
                                        raise TypeError(
                                            f"Parameter '{param_name}' kind mismatch. "
                                            f"Parent: {parent_param.kind}, Child: {child_param.kind}"
                                        )
                                    if not self._is_second_stronger_first(
                                            child_param.annotation,
                                            parent_param.annotation
                                        ):
                                        raise TypeError(
                                            f"Parent parameter annotation ({parent_param.annotation}) is weaker than "
                                            f"child paremeter annotation {child_param.annotation} for '{param_name}' argument. "
                                        )
                            
                            if not self._is_second_stronger_first(
                                    parent_sig.return_annotation,
                                    child_sig.return_annotation
                                ):
                                raise TypeError(
                                    f"Parent method return annotation {parent_sig.return_annotation} is stronger than "
                                    f"child method return annotation {child_sig.return_annotation} for '{var.name}' method. "
                                )
                    else:
                        safe_methods = ["__init__", "__abstractmethods__"]
                        if var.name in safe_methods:
                            continue
                        if not issubclass(type(child_var.value), Decorator):
                            raise TypeError(
                                f"Overriden method {var.name} must contains Decorator value, "
                                "because base class method isn't abstract. "
                            )
                else:
                    if not issubclass(type(child_var.value), Descriptor) and not child_var.value is Descriptor and child_var.value != None:
                        raise TypeError(
                            f"Overriden attribute {var.name} is only allowed to have Descriptor value. "
                        )
                