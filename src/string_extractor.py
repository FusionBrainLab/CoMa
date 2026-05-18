import re
from typing import Dict, Any

from .core.base import Function

class StringExtractor(Function):
    def __init__(self, *, pattern: str) -> None:
        self.pattern = pattern

    def __call__(self, *, string: str) -> str:
        output = re.findall(self.pattern, string)[0]
        return output