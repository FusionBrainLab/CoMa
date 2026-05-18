from typing import List, Any, Dict

from .dataset_creator import DatasetCreator
from ..dataset_processor import DatasetProcessor

class PreprocessDatasetCreator(DatasetCreator):
    def __init__(self, *, base_creator: DatasetCreator,
                        preprocessor: DatasetProcessor) -> None:
        self.base_creator = base_creator
        self.preprocessor = preprocessor

    def __call__(self) -> Dict[str, List[Any]]:
        dataset = self.base_creator()
        dataset = self.preprocessor(dataset=dataset)
        return dataset
