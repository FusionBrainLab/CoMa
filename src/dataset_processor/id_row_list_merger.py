from typing import List, Any, Dict

import pandas as pd

from .dataset_processor import DatasetProcessor

class IdRowListMerger(DatasetProcessor):
    def __init__(self, *, id_col: str,
                        list_cols: List[str]) -> None:
        self.id_col = id_col
        self.list_cols = list_cols
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        df = pd.DataFrame(dataset)
        rows = []
        for _, group in df.groupby(self.id_col, sort=False, dropna=False):
            row = group.iloc[0].to_dict()
            for col in self.list_cols:
                row[col] = []
                for value in group[col]:
                    if not isinstance(value, list):
                        continue
                    for item in value:
                        if item not in row[col]:
                            row[col].append(item)
            rows.append(row)

        new_df = pd.DataFrame(rows, columns=df.columns)
        new_dataset = new_df.to_dict("list")
        return new_dataset