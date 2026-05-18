from abc import ABC, abstractmethod
from typing import Any, Protocol, Dict

from ..core.base import Container

class StatefulProtocol(Protocol):
    def state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...

class TorchStateManager(Container, ABC):
    @abstractmethod
    def set_state(self, *, statefuls: Dict[str, StatefulProtocol]) -> None:
        ...
    
    @abstractmethod
    def get_state(self, *, statefuls: Dict[str, StatefulProtocol]) -> Dict[str, Any]:
        ...