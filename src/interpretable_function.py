from typing import Any

from .core.base import Function

class InterpretableFunction(Function):
    function: str
    def __init__(self, *, function: str,
                        returns_output: bool) -> None:
        self.function = function
        self.returns_output = returns_output
        
    def __call__(self, **kwargs: Any) -> Any:
        if self.returns_output:
            return eval(self.function, kwargs)
        else:
            return exec(self.function, kwargs)
