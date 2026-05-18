from abc import ABC, abstractmethod
from typing import Any, Dict
import json

from .serializer import Serializer
from ..string_formatter import StringFormatter

class JsonSerializer(Serializer):
    def __call__(self, *, obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)