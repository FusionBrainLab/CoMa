from typing import List, Any, Dict

import pandas as pd

from .dataset_processor import DatasetProcessor

class IdRowMeanMerger(DatasetProcessor):
    def __init__(self, *, id_col: str,
                        mean_cols: List[str]) -> None:
        self.id_col = id_col
        self.mean_cols = mean_cols

    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        df = pd.DataFrame(dataset)
        rows = []
        for _, group in df.groupby(self.id_col, sort=False, dropna=False):
            row = group.iloc[0].to_dict()
            for col in self.mean_cols:
                row[col] = group[col].mean()
            rows.append(row)

        new_df = pd.DataFrame(rows, columns=df.columns)
        return new_df.to_dict("list")
