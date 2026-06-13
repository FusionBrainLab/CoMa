from typing import Any, Dict, Literal
import re

from .string_deserializer import StringDeserializer

class RegexArgsDeserializer(StringDeserializer):
    def __init__(self, *, args_expressions: Dict[str, str],
                        numeric_args: Dict[str, Literal["int", "float"]]) -> None:
        self.args_expressions = args_expressions
        self.numeric_args = numeric_args

    def __call__(self, *, string: str) -> Any:
        args = {}
        for key, expression in self.args_expressions.items():
            match = re.search(expression, string)
            assert match is not None
            if key in self.numeric_args:
                if self.numeric_args[key] == "int":
                    args[key] = int(match.group(1))
                elif self.numeric_args[key] == "float":
                    args[key] = float(match.group(1))
                else:
                    raise ValueError(f"Invalid numeric argument type: {self.numeric_args[key]}")
            else:
                args[key] = match.group(1)
        return args