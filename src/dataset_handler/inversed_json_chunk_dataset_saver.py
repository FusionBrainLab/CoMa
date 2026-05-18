from typing import List, Any, Dict
import json
import math
import os

from tqdm import tqdm

from .dataset_handler import DatasetHandler

class InversedJsonChunkDatasetSaver(DatasetHandler):
    def __init__(self, *, folder_path: str,
                        chunk_length: int) -> None:
        self.folder_path = folder_path
        self.chunk_length = chunk_length

    def __call__(self, *, dataset: Dict[str, List[Any]]) -> None:
        inversed_dataset = [{k: dataset[k][i] for k in dataset.keys()} for i in range(len(dataset[list(dataset.keys())[0]]))]
        n_chunks = math.ceil(len(inversed_dataset)/self.chunk_length)
        chunks = [inversed_dataset[i * self.chunk_length: min((i + 1) * self.chunk_length, len(inversed_dataset))] for i in range(n_chunks)]
        for i, c in tqdm(enumerate(chunks), total=len(chunks)):
            path = os.path.join(self.folder_path, f"{i}.json")
            with open(path, "w+") as f:
                json.dump(c, f, ensure_ascii=False)