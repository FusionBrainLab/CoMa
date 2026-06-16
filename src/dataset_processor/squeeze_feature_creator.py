from typing import List, Any, Dict

import pandas as pd

from .dataset_processor import DatasetProcessor

class SqueezeFeatureCreator(DatasetProcessor):
    def __init__(self, *, cols: List[str],
                        fixed_feature_values: Dict[Any, Dict[str, Any]],
                        fixed_feature_col: str,
                        value_feature_col: str) -> None:
        self.cols = cols
        self.fixed_feature_values = fixed_feature_values
        self.fixed_feature_col = fixed_feature_col
        self.value_feature_col = value_feature_col
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        new_dataset = {k: [] for k in dataset.keys()}
        new_dataset[self.fixed_feature_col] = []
        new_dataset[self.value_feature_col] = []
        
        for i in range(len(next(iter(dataset.values())))):
            for group_value, fixed_values in self.fixed_feature_values.items():
                if all(dataset[col][i] == value for col, value in fixed_values.items()):
                    for col in new_dataset.keys():
                        if col not in [self.fixed_feature_col, self.value_feature_col]:
                            new_dataset[col].append(dataset[col][i])
                    new_dataset[self.fixed_feature_col].append(group_value)
                    new_dataset[self.value_feature_col].append(next(dataset[col][i] for col in self.cols if col not in fixed_values))
        return new_dataset