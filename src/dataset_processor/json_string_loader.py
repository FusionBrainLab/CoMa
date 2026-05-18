from typing import List, Any, Dict
import json

import pandas as pd

from .dataset_processor import DatasetProcessor

class JsonStringLoader(DatasetProcessor):
    def __init__(self, *, cols: List[str]) -> None:
        self.cols = cols
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        df = pd.DataFrame(dataset)
        for col in self.cols:
            df[col] = df.apply(lambda row: json.loads(row[col]), axis=1)
        new_dataset = df.to_dict("list")
        return new_dataset