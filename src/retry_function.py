from typing import Any, Dict

from .core.base import Function

class RetryFunction(Function):
    base_function: Function
    def __init__(self, *, base_function: Function,
                        n_retries: int,
                        default_result: Any) -> None:
        self.base_function = base_function
        self.n_retries = n_retries
        self.default_result = default_result

    def __call__(self, **kwargs: Any) -> Any:
        counter = self.n_retries
        while counter > 0:
            try:
                return self.base_function(**kwargs)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"RetryFunction handled an exception: {repr(e)}")
                counter -= 1
        return self.default_result