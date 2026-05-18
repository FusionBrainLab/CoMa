from typing import List, Any, Dict

from .dataset_processor import DatasetProcessor

class CompositionProcessor(DatasetProcessor):
    def __init__(self, *, base_processors: List[DatasetProcessor]) -> None:
        self.base_processors = base_processors
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        new_dataset = dataset
        for processor in self.base_processors:
            new_dataset = processor(dataset=new_dataset)
        return new_dataset