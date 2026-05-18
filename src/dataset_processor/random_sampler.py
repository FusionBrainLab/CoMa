from typing import List, Any, Dict
import random

import pandas as pd

from .dataset_processor import DatasetProcessor

class RandomSampler(DatasetProcessor):
    def __init__(self, *, n_samples: int) -> None:
        self.n_samples = n_samples
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        """print(len(pd_dataset))
        raise"""
        inds = list(range(len(pd_dataset)))
        sample_inds = random.sample(inds, self.n_samples)
        pd_dataset = pd_dataset.iloc[sample_inds]
        dataset = pd_dataset.to_dict("list")
        return dataset