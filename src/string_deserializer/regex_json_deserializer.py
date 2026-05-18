from abc import ABC, abstractmethod
from typing import Any
import re
import json

from .string_deserializer import StringDeserializer

class RegexJsonDeserializer(StringDeserializer):
    def __init__(self, *, pattern: str) -> None:
        self.pattern = pattern

    def __call__(self, *, string: str) -> Any:
        extracted = re.findall(self.pattern, string)[0]
        data = json.loads(extracted)
        return data
