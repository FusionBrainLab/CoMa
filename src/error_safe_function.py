from typing import Any, Dict

from .core.base import Function

class ErrorSafeFunction(Function):
    base_function: Function
    def __init__(self, *, base_function: Function,
                        default_result: Any) -> None:
        self.base_function = base_function
        self.default_result = default_result
    def __call__(self, **kwargs: Any) -> Any:
        try:
            return self.base_function(**kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            """if "curve and relativeCurve must have the same length" in repr(e):
                print(kwargs)
                raise"""
            #raise
            print(f"ErrorSafeFunction handled an exception: {repr(e)}")
            return self.default_result