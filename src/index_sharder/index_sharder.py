from abc import ABC, abstractmethod
from typing import List

from ..core.base import Function

class IndexSharder(Function, ABC):
    @abstractmethod
    def __call__(self, *, n_indexes: int, n_shards: int) -> List[List[int]]:
        ...