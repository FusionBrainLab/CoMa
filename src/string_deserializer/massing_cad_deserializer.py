from abc import ABC, abstractmethod
from typing import Any
import re
import json

from .string_deserializer import StringDeserializer

class MassingCADDeserializer(StringDeserializer):
    def __init__(self, *, template: str,
                        separator: str) -> None:
        self.template = template
        self.separator = separator

    def __call__(self, *, string: str) -> Any:
        field_names = ["id", "massing"]

        first_pattern = self.template
        for field in field_names:
            first_pattern = first_pattern.replace('{' + field + '}', r'(?:.|\n)*?')
        chunks = re.findall(first_pattern, string)

        massing = []
        for c in chunks:
            pattern = self.template
    
            # Replace each format specifier with a regex capture group
            for field in field_names:
                pattern = pattern.replace('{' + field + '}', r'((?:.|\n)*?)')

            # Match the pattern against the formatted string
            match = re.match(pattern, c)
            assert match != None, "Can't deserialize massing cad from string. "
            massing.append(dict(zip(field_names, match.groups())))
        return massing
                