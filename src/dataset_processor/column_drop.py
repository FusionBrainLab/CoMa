from typing import List, Any, Dict

import pandas as pd

from .dataset_processor import DatasetProcessor

class ColumnDrop(DatasetProcessor):
    def __init__(self, *, cols: List[str]) -> None:
        self.cols = cols
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        new_dataset = {k: dataset[k] for k in dataset.keys() if k not in self.cols}
        return new_dataset