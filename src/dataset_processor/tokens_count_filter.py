from typing import List, Any, Dict
import random

import pandas as pd
from tqdm import tqdm

from .dataset_processor import DatasetProcessor
from ..memory_tokenizer import MemoryTokenizer

class TokensCountFilter(DatasetProcessor):
    def __init__(self, *, memory_col: str,
                        memory_tokenizer: MemoryTokenizer,
                        ids_key: str,
                        max_tokens: int) -> None:
        self.memory_col = memory_col
        self.memory_tokenizer = memory_tokenizer
        self.ids_key = ids_key
        self.max_tokens = max_tokens
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        tqdm.pandas()
        pd_dataset["n_tokens"] = pd_dataset.progress_apply(lambda row: self.memory_tokenizer(memories=[row[self.memory_col]])[self.ids_key].shape[1], axis=1)
        pd_dataset = pd_dataset[pd_dataset["n_tokens"] <= self.max_tokens].drop(columns=["n_tokens"])
        dataset = pd_dataset.to_dict("list")
        return dataset