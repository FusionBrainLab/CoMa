from typing import List, Any, Dict
import json

from .dataset_handler import DatasetHandler

class InversedJsonDatasetSaver(DatasetHandler):
    def __init__(self, *, path: str) -> None:
        self.path = path

    def __call__(self, *, dataset: Dict[str, List[Any]]) -> None:
        inversed_dataset = [{k: dataset[k][i] for k in dataset.keys()} for i in range(len(dataset[list(dataset.keys())[0]]))]
        with open(self.path, "w+") as f:
            json.dump(inversed_dataset, f, ensure_ascii=False)