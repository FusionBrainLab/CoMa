from abc import ABC, abstractmethod
from typing import Any
import re
import json

from rosreestr2coord.parser import Area

from .string_deserializer import StringDeserializer

class DummyRosreestrIdParser(StringDeserializer):
    def __init__(self, *, default_output_path: str) -> None:
        self.default_output_path = default_output_path

    def __call__(self, *, string: str) -> Any:
        with open(self.default_output_path, 'r') as f:
            return json.load(f)