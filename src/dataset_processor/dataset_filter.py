from typing import List, Any, Dict

import pandas as pd

from .dataset_processor import DatasetProcessor
from ..protocols import IndicatorProtocol

class DatasetFilter(DatasetProcessor):
    def __init__(self, *, filter: IndicatorProtocol,
                        cols_to_args_mapping: Dict[str, str]) -> None:
        self.filter = filter
        self.cols_to_args_mapping = cols_to_args_mapping
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        keys = list(dataset.keys())
        length = len(dataset[keys[0]])

        pd_dataset = pd.DataFrame(dataset)
        pd_dataset = pd_dataset[pd_dataset.apply(lambda row: self.filter(**{self.cols_to_args_mapping[k]: row[k] for k in self.cols_to_args_mapping.keys()}), axis=1)]
        new_dataset = pd_dataset.to_dict("list")

        return new_dataset