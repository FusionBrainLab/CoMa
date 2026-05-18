from typing import List, Any, Dict

import pandas as pd
from tqdm import tqdm

from .dataset_processor import DatasetProcessor
from ..core.base import Function

class FunctionProcessor(DatasetProcessor):
    def __init__(self, *, function: Function,
                        cols_to_args_mapping: Dict[str, str],
                        output_key: str) -> None:
        self.function = function
        self.cols_to_args_mapping = cols_to_args_mapping
        self.output_key = output_key
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        tqdm.pandas()
        
        #dataset = {k: v for k, v in dataset.items() if k in self.cols_to_args_mapping.keys()}
        
        pd_dataset = pd.DataFrame(dataset)
        pd_dataset[self.output_key] = pd_dataset.progress_apply(lambda row: self.function(**{self.cols_to_args_mapping[k]: row[k] for k in self.cols_to_args_mapping.keys()}), axis=1)
        dataset = pd_dataset.to_dict("list")
        return dataset