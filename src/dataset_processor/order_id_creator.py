from typing import List, Any, Dict

from .dataset_processor import DatasetProcessor

class OrderIdCreator(DatasetProcessor):
    def __init__(self, *, field_name: str) -> None:
        self.field_name = field_name
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        dataset[self.field_name] = []
        for i in range(len(dataset[list(dataset.keys())[0]])):
            dataset[self.field_name].append(str(i))
        return dataset