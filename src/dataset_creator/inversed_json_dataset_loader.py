from typing import List, Any, Dict
import json
import os
import re

from .dataset_creator import DatasetCreator

class InversedJsonDatasetLoader(DatasetCreator):
    def __init__(self, *, path: str) -> None:
        self.path = path
    
    def __call__(self) -> Dict[str, List[Any]]:
        with open(self.path, "r") as f:
            data = json.load(f)
        keys = list(data[0].keys())
        dataset = {k: [data[i][k] for i in range(len(data))] for k in keys}
        return dataset
    