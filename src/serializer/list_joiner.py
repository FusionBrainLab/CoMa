from abc import ABC, abstractmethod
from typing import Any, Dict

from .serializer import Serializer
from ..string_formatter import StringFormatter

class ListJoiner(Serializer):
    def __init__(self, *, separator: str) -> None:
        self.separator = separator

    def __call__(self, *, obj: Any) -> str:
        return self.separator.join(obj)