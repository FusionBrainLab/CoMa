from typing import List, Any, Dict

import pandas as pd
from tqdm import tqdm

from .dataset_processor import DatasetProcessor
from ..core.base import Function

class ErrorSafeFunctionProcessor(DatasetProcessor):
    def __init__(self, *, function: Function,
                        cols_to_args_mapping: Dict[str, str],
                        dataset_arg: str,
                        output_key: str,
                        invalid_indicator_col: str) -> None:
        self.function = function
        self.cols_to_args_mapping = cols_to_args_mapping
        self.dataset_arg = dataset_arg
        self.output_key = output_key
        self.invalid_indicator_col = invalid_indicator_col
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        tqdm.pandas()
        def safe_function(row):
            try:
                return self.function(**{self.dataset_arg: dataset, **{self.cols_to_args_mapping[k]: row[k] for k in self.cols_to_args_mapping.keys()}})
            except:
                return "INVALID"
        pd_dataset[self.output_key] = pd_dataset.progress_apply(lambda row: safe_function(row), axis=1)
        pd_dataset[self.invalid_indicator_col] = pd_dataset.apply(lambda row: row[self.output_key] == "INVALID", axis=1)
        dataset = pd_dataset.to_dict("list")
        return dataset