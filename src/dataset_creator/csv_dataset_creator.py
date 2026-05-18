from typing import List, Any, Dict

import pandas as pd

from .dataset_creator import DatasetCreator

class CsvDatasetCreator(DatasetCreator):
    def __init__(self, *, path: str,
                        pandas_args: Dict[str, any]) -> None:
        self.path = path
        self.pandas_args = pandas_args

    def __call__(self) -> Dict[str, List[Any]]:
        dataset = pd.read_csv(self.path, **self.pandas_args).to_dict("list")
        return dataset
