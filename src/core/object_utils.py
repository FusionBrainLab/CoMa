from dataclasses import dataclass
from typing import Any, List, get_type_hints, Dict
import inspect
from copy import copy

@dataclass
class VarConfig:
    name: str
    type: type
    value: Any

"Неявно определено"
class BaseClassNamespaceParser():
    def __call__(self, cls: type) -> List[VarConfig]:
        base_vars = dir(cls)
        direct_vars = DirectClassNamespaceParser()(cls)
        direct_names = [c.name for c in direct_vars]
        inherited_vars = InheritedClassNamespaceParser()(cls)
        inherited_names = [c.name for c in inherited_vars]
        namespace = []
        for name in base_vars:
            if name not in direct_names and name not in inherited_names:
                value = getattr(cls, name, None)
                var_config = VarConfig(name, inspect.Parameter.empty, value)
                namespace.append(var_config)
        return namespace

"Явно определено в конкретном классе"
class DirectClassNamespaceParser():
    skip_vars: List[str] = ["__module__", "__qualname__", "__doc__", "__annotations__", "__weakref__", "__dict__"]
    def __call__(self, cls: type) -> List[VarConfig]:
        class_vars = vars(cls)
        annotations = class_vars.get("__annotations__", {})
        namespace = []
        var_names = []
        for name, value in class_vars.items():
            if name in self.skip_vars:
                continue
            var_names.append(name)
            default_type = inspect.Parameter.empty
            if inspect.isfunction(value) or inspect.ismethod(value):
                default_type = type(value)
            var_type = annotations.get(name, default_type)
            var_config = VarConfig(name, var_type, value)
            namespace.append(var_config)
        for name, var_type in annotations.items():
            if name not in var_names:
                var_config = VarConfig(name, var_type, None)
                namespace.append(var_config)
        return namespace

"Явно определено в родительских классах"
class InheritedClassNamespaceParser():
    skip_vars: List[str] = ["__module__", "__qualname__", "__doc__", "__annotations__", "__weakref__", "__dict__"]   
    def __call__(self, cls: type) -> List[VarConfig]:
        class_vars = {}
        for base in cls.__mro__:
            base_vars = base.__dict__
            for name, v in base_vars.items():
                if name not in class_vars:
                    class_vars[name] = v
        annotations = get_type_hints(cls)
        total_namespace = []
        var_names = []
        for name, value in class_vars.items():
            if name in self.skip_vars:
                continue
            var_names.append(name)
            default_type = inspect.Parameter.empty
            if inspect.isfunction(value) or inspect.ismethod(value):
                default_type = type(value)
            var_type = annotations.get(name, default_type)
            var_config = VarConfig(name, var_type, value)
            total_namespace.append(var_config)
        for name, var_type in annotations.items():
            if name not in var_names:
                var_config = VarConfig(name, var_type, None)
                total_namespace.append(var_config)
        direct_vars = DirectClassNamespaceParser()(cls)
        direct_names = [c.name for c in direct_vars]
        object_vars = dir(object)
        namespace = [c for c in total_namespace if c.name not in direct_names and c.name not in object_vars]
        return namespace
    
"Полное пространство имен"
class TotalClassNamespaceParser():
    def __call__(self, cls: type) -> List[VarConfig]:
        base_namespace = BaseClassNamespaceParser()(cls)
        base_names = {c.name: c for c in base_namespace}
        direct_namespace = DirectClassNamespaceParser()(cls)
        direct_names = {c.name: c for c in direct_namespace}
        inherited_namespace = InheritedClassNamespaceParser()(cls)
        inherited_names = {c.name: c for c in inherited_namespace}
        name_groups = [direct_names, inherited_names, base_names]
        namespace = []
        names = []
        for names_group in name_groups:
            for name, c in names_group.items():
                if name not in names:
                    namespace.append(c)
                    names.append(name)
        return namespace
 
class DirectClassDictNamespaceParser():
    skip_vars: List[str] = ["__module__", "__qualname__", "__doc__", "__annotations__", "__weakref__", "__dict__"]
    def __call__(self, namespace: Dict[str, Any]) -> List[VarConfig]:
        class_vars = namespace
        annotations = class_vars.get("__annotations__", {})
        namespace = []
        var_names = []
        for name, value in class_vars.items():
            if name in self.skip_vars:
                continue
            var_names.append(name)
            default_type = inspect.Parameter.empty
            if inspect.isfunction(value) or inspect.ismethod(value):
                default_type = type(value)
            var_type = annotations.get(name, default_type)
            var_config = VarConfig(name, var_type, value)
            namespace.append(var_config)
        for name, var_type in annotations.items():
            if name not in var_names:
                var_config = VarConfig(name, var_type, None)
                namespace.append(var_config)
        return namespace