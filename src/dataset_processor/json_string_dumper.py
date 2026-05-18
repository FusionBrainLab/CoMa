from typing import List, Any, Dict
import json

import pandas as pd
from tqdm import tqdm

from .dataset_processor import DatasetProcessor

class JsonStringDumper(DatasetProcessor):
    def __init__(self, *, cols: List[str]) -> None:
        self.cols = cols
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        df = pd.DataFrame(dataset)
        tqdm.pandas()
        for col in self.cols:
            df[col] = df.progress_apply(lambda row: json.dumps(row[col], ensure_ascii=False), axis=1)
        new_dataset = df.to_dict("list")
        return new_dataset