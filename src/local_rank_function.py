from typing import Any
import os

from .core.base import Function

class LocalRankFunction(Function):
    function: Function
    rank: int
    def __init__(self, *, function: Function, rank: int) -> None:
        self.function = function
        self.rank = rank
    
    def __call__(self, **kwargs: Any) -> Any:
        local_rank = int(os.environ['LOCAL_RANK'])
        if local_rank == self.rank:
            return self.function(**kwargs)
