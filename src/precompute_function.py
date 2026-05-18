from typing import Any, Dict

import pandas as pd

from .core.base import Function
from .dataset_creator import DatasetCreator

class PrecomputeFunction(Function):
    def __init__(self, *, base_function: Function,
                        precompute_loader: DatasetCreator,
                        id_key: str,
                        result_col: str) -> None:
        self.base_function = base_function
        self.precompute_loader = precompute_loader
        self.id_key = id_key
        self.result_col = result_col

        precomputes = pd.DataFrame(precompute_loader())
        precomputes["id_buffer"] = precomputes[id_key]
        precomputes = precomputes.set_index("id_buffer")
        self.precomputes = precomputes

    def __call__(self, **kwargs: Any) -> Any:
        if kwargs[self.id_key] in self.precomputes[self.id_key]:
            return self.precomputes.loc[kwargs[self.id_key]][self.result_col]
        else:
            return self.base_function(**kwargs)