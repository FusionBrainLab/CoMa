from typing import List, Any, Dict
import json
import os
import re

from tqdm import tqdm

from .dataset_creator import DatasetCreator
from ..dataset_processor import DatasetProcessor

class InversedJsonChunkPreprocessDatasetLoader(DatasetCreator):
    def __init__(self, *, folder_path: str,
                        name_pattern: str,
                        verbose: bool,
                        preprocessor: DatasetProcessor) -> None:
        self.folder_path = folder_path
        self.name_pattern = name_pattern
        self.verbose = verbose
        self.preprocessor = preprocessor
    
    def __call__(self) -> Dict[str, List[Any]]:
        names = os.listdir(self.folder_path)
        valid_names = [n for n in names if re.fullmatch(self.name_pattern, n)]

        dataset = None
        cycle = tqdm(valid_names) if self.verbose else valid_names
        for name in cycle:
            path = os.path.join(self.folder_path, name)
            with open(path, "r") as f:
                data = json.load(f)
            keys = list(data[0].keys())
            chunk = {k: [data[i][k] for i in range(len(data))] for k in keys}

            processed_chunk = self.preprocessor(dataset=chunk)
            del chunk

            if dataset == None:
                dataset = processed_chunk
            else:
                for k in keys:
                    dataset[k].extend(processed_chunk[k])

        return dataset