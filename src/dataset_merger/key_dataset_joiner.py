from typing import Any, Dict, List, Optional

import pandas as pd

from .dataset_merger import DatasetMerger

class KeyDatasetJoiner(DatasetMerger):
    """Horizontally join named datasets on a shared key column."""

    def __init__(self, *, key_col: str, 
                        how: str,
                        dataset_cols: Optional[Dict[str, List[str]]] = None) -> None:
        self.key_col = key_col
        self.how = how
        self.dataset_cols = dataset_cols

    def __call__(self, *, datasets: Dict[str, Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
        if not datasets:
            return {}

        merged: pd.DataFrame | None = None
        for name, dataset in datasets.items():
            cols = list(dataset.keys())
            if self.dataset_cols and name in self.dataset_cols:
                cols = [self.key_col] + [
                    col
                    for col in self.dataset_cols[name]
                    if col != self.key_col
                ]
            frame = pd.DataFrame({col: dataset[col] for col in cols})
            if merged is None:
                merged = frame
                continue
            # On column conflicts keep the left name and suffix the incoming side
            # with the dataset key so values from both sources are preserved.
            merged = merged.merge(
                frame,
                on=self.key_col,
                how=self.how,
                suffixes=("", f"_{name}"),
            )

        assert merged is not None
        return merged.to_dict("list")
