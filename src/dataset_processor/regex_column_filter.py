from typing import List, Any, Dict

import pandas as pd

from .dataset_processor import DatasetProcessor

class RegexColumnFilter(DatasetProcessor):
    def __init__(self, *, col: str,
                        pattern: str) -> None:
        self.col = col
        self.pattern = pattern

    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        df = pd.DataFrame(dataset)
        df = df[df[self.col].str.contains(self.pattern, regex=True, na=False)]
        return df.to_dict("list")
