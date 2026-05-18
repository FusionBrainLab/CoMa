from typing import List, Any, Dict

import pandas as pd

from .dataset_processor import DatasetProcessor

class ColumnMergeAugmenter(DatasetProcessor):
    def __init__(self, *, merge_cols: List[str],
                        new_col: str) -> None:
        self.merge_cols = merge_cols
        self.new_col = new_col
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        new_datasets = []
        for col in self.merge_cols:
            new_dataset = pd_dataset.copy()
            new_dataset[self.new_col] = new_dataset[col]
            new_dataset = new_dataset.drop(columns=self.merge_cols)
            new_datasets.append(new_dataset)
        new_dataset = pd.concat(new_datasets)
        new_dataset = new_dataset.to_dict("list")
        return new_dataset