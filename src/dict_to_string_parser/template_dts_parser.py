from typing import List, Any, Dict

from .dict_to_string_parser import DictToStringParser
from ..string_formatter import StringFormatter

class TemplateDTSParser(DictToStringParser):
    def __init__(self, *, template: str) -> None:
        self.template = template
        self.string_formatter = StringFormatter()

    def __call__(self, *, dictionary: Dict[str, Any]) -> str:
        output = self.string_formatter(string=self.template, args=dictionary)
        return output
