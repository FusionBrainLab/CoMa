import re
from typing import Dict, Any

from .core.base import Function

class StringFormatter(Function):
    def __call__(self, *, string: str, args: Dict[str, Any]) -> str:
        output = string
        for key, value in args.items():
            pattern = "{" + key + "}"
            try:
                output = re.sub(pattern, str(value), output)
            except:
                continue
        return output
