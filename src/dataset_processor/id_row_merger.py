from typing import List, Any, Dict

import pandas as pd

from .dataset_processor import DatasetProcessor

class IdRowMerger(DatasetProcessor):
    def __init__(self, *, id_col: str) -> None:
        self.id_col = id_col
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        df = pd.DataFrame(dataset)
        new_df = df.groupby(self.id_col).agg(list)
        new_dataset = new_df.to_dict("list")
        return new_dataset