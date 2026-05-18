from typing import List, Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset_processor import DatasetProcessor

class SplitProcessor(DatasetProcessor):
    def __init__(self, *, split_size: float,
                        random_seed: int) -> None:
        self.split_size = split_size
        self.random_seed = random_seed
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        _, split = train_test_split(pd_dataset, test_size=self.split_size, random_state=self.random_seed)
        dataset = split.to_dict("list")
        return dataset
