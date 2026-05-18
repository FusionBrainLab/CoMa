from typing import List, Any, Dict

from .dataset_creator import DatasetCreator
from ..dataset_processor import DatasetProcessor

class PartialDatasetLoader(DatasetCreator):
    def __init__(self, *, base_creator: DatasetCreator,
                        dataset_range: List[int]) -> None:
        self.base_creator = base_creator
        self.dataset_range = dataset_range

    def __call__(self) -> Dict[str, List[Any]]:
        dataset = self.base_creator()
        final_dataset = {k: dataset[k][self.dataset_range[0]: self.dataset_range[1]] for k in dataset.keys()}
        return final_dataset