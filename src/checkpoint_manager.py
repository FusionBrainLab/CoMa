from typing import Any, Dict, Protocol, runtime_checkable
import os

from .core.base import Function
from .storage import Storage
from .protocols import IndicatorProtocol, StringCreatorProtocol

@runtime_checkable
class CheckpointableProtocol(Protocol):
    def load_checkpoint(self, *, checkpoint: object) -> None:
        ...
    
    def save_checkpoint(self) -> object:
        ...

class CheckpointManager(Function):
    storage: Storage
    load_condition_checker: IndicatorProtocol
    save_condition_checker: IndicatorProtocol
    load_path_creator: StringCreatorProtocol
    save_path_creator: StringCreatorProtocol
    def __init__(self, *, storage: Storage,
                    load_condition_checker: IndicatorProtocol,
                    save_condition_checker: IndicatorProtocol,
                    load_path_creator: StringCreatorProtocol,
                    save_path_creator: StringCreatorProtocol) -> None:
        self.storage = storage
        self.load_condition_checker = load_condition_checker
        self.save_condition_checker = save_condition_checker
        self.load_path_creator = load_path_creator
        self.save_path_creator = save_path_creator
        
    def __call__(self, *, checkpointable: CheckpointableProtocol, conditions: Dict[str, Any]) -> None:
        if self.load_condition_checker(**conditions):
            path = self.load_path_creator(**conditions)
            assert os.path.exists(path), "Load path doesn't exist. "
            self.storage.set_path(path=path)
            checkpoint = self.storage.load()
            checkpointable.load_checkpoint(checkpoint=checkpoint)
        if self.save_condition_checker(**conditions):
            checkpoint = checkpointable.save_checkpoint()
            path = self.save_path_creator(**conditions)
            os.makedirs(path, exist_ok=True)
            self.storage.set_path(path=path)
            self.storage.save(obj=checkpoint)