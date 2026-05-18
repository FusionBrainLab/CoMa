from typing import Any, List, Dict
from dataclasses import dataclass

from .storage import Storage

@dataclass
class DictStorageConfig:
    object_path: List[str]
    storage: Storage

class StorageDict(Storage):
    storage_configs: List[DictStorageConfig]
    def __init__(self, *, storage_configs: List[DictStorageConfig]) -> None:
        self.storage_configs = storage_configs
    
    def set_path(self, *, path: str) -> None:
        for config in self.storage_configs:
            config.storage.set_path(path=path)
    
    def load(self) -> object:
        obj = {}
        for config in self.storage_configs:
            subobj = config.storage.load()
            buffer = obj
            for step in config.object_path[:-1]:
                if step not in buffer:
                    buffer[step] = {}
                buffer = buffer[step]
            buffer[config.object_path[-1]] = subobj
        return obj
    
    def save(self, *, obj: object) -> None:
        assert isinstance(obj, dict)
        for config in self.storage_configs:
            subobj = obj
            for step in config.object_path:
                subobj = subobj[step]
            config.storage.save(obj=subobj)