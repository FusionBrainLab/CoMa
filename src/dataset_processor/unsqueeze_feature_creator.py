from typing import List, Any, Dict

import pandas as pd

from .dataset_processor import DatasetProcessor

class UnsqueezeFeatureCreator(DatasetProcessor):
    def __init__(self, *, feature_values: List[Dict[str, Any]],
                        new_feature_col: str,
                        old_feature_value: Any,
                        new_feature_value: Any) -> None:
        self.feature_values = feature_values
        self.new_feature_col = new_feature_col
        self.old_feature_value = old_feature_value
        self.new_feature_value = new_feature_value
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        new_dataset = {k: [] for k in dataset.keys()}
        new_dataset[self.new_feature_col] = []
        
        for i in range(len(next(iter(dataset.values())))):
            for col in dataset.keys():
                new_dataset[col].append(dataset[col][i])
            new_dataset[self.new_feature_col].append(self.old_feature_value)
            
            for feature_values in self.feature_values:
                if all(dataset[col][i] == value for col, value in feature_values.items()):
                    for col in dataset.keys():
                        new_dataset[col].append(dataset[col][i])
                    new_dataset[self.new_feature_col].append(self.new_feature_value)
        return new_dataset