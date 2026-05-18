from typing import List, Any, Dict

import pandas as pd

from .dataset_creator import DatasetCreator

class MemoryDatasetLoader(DatasetCreator):
    def __init__(self, *, dataset: Dict[str, List[Any]]) -> None:
        self.dataset = dataset

    def __call__(self) -> Dict[str, List[Any]]:
        return self.dataset