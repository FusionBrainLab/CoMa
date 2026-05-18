from abc import ABC, abstractmethod
from typing import Any, Dict

from .serializer import Serializer
from ..string_formatter import StringFormatter

class ArgsListSerializer(Serializer):
    def __init__(self, *, template: str,
                        separator: str,
                        args_serializers: Dict[str, Serializer]) -> None:
        self.template = template
        self.separator = separator
        self.args_serializers = args_serializers
        self.string_formatter = StringFormatter()

    def __call__(self, *, obj: Any) -> str:
        output = []
        for m in obj:
            args = {}
            for k, v in m.items():
                if k in self.args_serializers:
                    args[k] = self.args_serializers[k](obj=v)
                else:
                    args[k] = str(v)
            chunk = self.string_formatter(string=self.template, args=args)
            output.append(chunk)
        output = self.separator.join(output)
        return output