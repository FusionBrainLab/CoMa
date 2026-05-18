from typing import List, Any, Dict

from .dataset_handler import DatasetHandler
from ..dataset_processor import DatasetProcessor

class PreprocessDatasetHandler(DatasetHandler):
    def __init__(self, *, base_handler: DatasetHandler,
                        preprocessor: DatasetProcessor) -> None:
        self.base_saver = base_handler
        self.preprocessor = preprocessor

    def __call__(self, *, dataset: Dict[str, List[Any]]) -> None:
        dataset = self.preprocessor(dataset=dataset)
        self.base_saver(dataset=dataset)