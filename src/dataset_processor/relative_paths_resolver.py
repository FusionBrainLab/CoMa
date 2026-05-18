from typing import List, Any, Dict
import os

import pandas as pd

from .dataset_processor import DatasetProcessor

class RelativePathsResolver(DatasetProcessor):
    def __init__(self, *, base_path: str,
                        cols: List[str]) -> None:
        self.base_path = base_path
        self.cols = cols
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        for col in self.cols:
            pd_dataset[col] = pd_dataset.apply(lambda row: os.path.join(self.base_path, row[col]), axis=1)
        dataset = pd_dataset.to_dict("list")
        return dataset