from typing import Protocol, Any, Dict

class IndicatorProtocol(Protocol):
    def __call__(self, **kwargs: Any) -> bool:
        ...

class StringCreatorProtocol(Protocol):
    def __call__(self, **kwargs: Any) -> str:
        ...

class DictCreatorProtocol(Protocol):
    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        ...