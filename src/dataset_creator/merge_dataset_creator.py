from typing import List, Any, Dict

from .dataset_creator import DatasetCreator
from ..dataset_merger import DatasetMerger

class MergeDatasetCreator(DatasetCreator):
    def __init__(self, *, base_creators: Dict[str, DatasetCreator],
                        merger: DatasetMerger) -> None:
        self.base_creators = base_creators
        self.merger = merger

    def __call__(self) -> Dict[str, List[Any]]:
        datasets = {name: creator() for name, creator in self.base_creators.items()}
        return self.merger(datasets=datasets)
