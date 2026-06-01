from typing import List, Any, Dict
import random

import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel

from .dataset_processor import DatasetProcessor
from ..memory_tokenizer import MemoryTokenizer
from ..agent_message import AgentMessage

class ModalityEstimationTokensCountFilter(DatasetProcessor):
    def __init__(self, *, memory_col: str,
                        memory_tokenizer: MemoryTokenizer,
                        ids_key: str,
                        max_tokens: int,
                        modality_estimations: Dict[str, int],
                        parallelize: int) -> None:
        self.memory_col = memory_col
        self.memory_tokenizer = memory_tokenizer
        self.ids_key = ids_key
        self.max_tokens = max_tokens
        self.modality_estimations = modality_estimations
        self.parallelize = parallelize
        if self.parallelize > 0:
            pandarallel.initialize(nb_workers=self.parallelize, progress_bar=True)
        else:
            pandarallel.initialize(nb_workers=1, progress_bar=True)
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        tqdm.pandas()
        def get_tokens_count(row):
            new_memory = []
            modality_tokens_count = 0
            for message in row[self.memory_col]:
                new_content = []
                for content in message.content:
                    if content.modality not in self.modality_estimations:
                        new_content.append(content)
                    else:
                        modality_tokens_count += self.modality_estimations[content.modality]
                new_message = AgentMessage(role=message.role, content=new_content)
                new_memory.append(new_message)
            base_tokens_count = self.memory_tokenizer(memories=[new_memory])[self.ids_key].shape[1]
            return base_tokens_count + modality_tokens_count
        pd_dataset["n_tokens"] = pd_dataset.parallel_apply(get_tokens_count, axis=1)
        pd_dataset = pd_dataset[pd_dataset["n_tokens"] <= self.max_tokens].drop(columns=["n_tokens"])
        dataset = pd_dataset.to_dict("list")
        return dataset