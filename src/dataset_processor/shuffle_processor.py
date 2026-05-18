from typing import List, Any, Dict
import random

import pandas as pd

from .dataset_processor import DatasetProcessor

class ShuffleProcessor(DatasetProcessor):
    def __init__(self, *, random_seed: int) -> None:
        self.random_seed = random_seed
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        shuffle_pd_dataset = pd_dataset.sample(frac=1, random_state=self.random_seed)
        shuffle_dataset = shuffle_pd_dataset.to_dict("list")
        return shuffle_dataset