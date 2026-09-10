from typing import List, Any, Dict
import json
import math
import os

from tqdm import tqdm
import pandas as pd

from .dataset_handler import DatasetHandler

class InversedJsonChunkTreeDatasetSaver(DatasetHandler):
    def __init__(self, *, folder_path: str,
                        chunk_length: int,
                        path_feature: str) -> None:
        self.folder_path = folder_path
        self.chunk_length = chunk_length
        self.path_feature = path_feature

    def __call__(self, *, dataset: Dict[str, List[Any]]) -> None:
        unique_path_features = list(set(dataset[self.path_feature]))
        pd_dataset = pd.DataFrame(dataset)
        for path_feature in tqdm(unique_path_features):
            local_pd_dataset = pd_dataset[
                pd_dataset[self.path_feature] == path_feature
            ]
            if len(local_pd_dataset) == 0:
                continue

            subfolder_path = os.path.join(self.folder_path, path_feature)
            os.makedirs(subfolder_path, exist_ok=True)
            local_dataset = local_pd_dataset.to_dict("list")
            inversed_local_dataset = [
                {
                    key: local_dataset[key][i]
                    for key in local_dataset.keys()
                }
                for i in range(len(local_pd_dataset))
            ]

            n_chunks = math.ceil(len(inversed_local_dataset) / self.chunk_length)
            chunks = [
                inversed_local_dataset[
                    i * self.chunk_length:
                    min((i + 1) * self.chunk_length, len(inversed_local_dataset))
                ]
                for i in range(n_chunks)
            ]
            for i, chunk in enumerate(chunks):
                path = os.path.join(subfolder_path, f"{i}.json")
                with open(path, "w+") as f:
                    json.dump(chunk, f, ensure_ascii=False)
