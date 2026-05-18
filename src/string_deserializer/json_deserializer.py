from abc import ABC, abstractmethod
from typing import Any
import re
import json

from .string_deserializer import StringDeserializer

class JsonDeserializer(StringDeserializer):
    def __call__(self, *, string: str) -> Any:
        return json.loads(string)