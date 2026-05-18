from typing import List, Any, Dict

from .metric import Metric
from ..dataset_processor import DatasetFilter

class FilterLength(Metric):
    def __init__(self, *, filter: DatasetFilter) -> None:
        self.filter = filter

    def __call__(self, *, submit: Dict[str, List[Any]]) -> float:
        dataset = self.filter(dataset=submit)
        keys = list(dataset.keys())
        length = len(dataset[keys[0]])
        return length