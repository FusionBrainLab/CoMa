from abc import ABC, abstractmethod
from typing import List, Any, Dict, Literal
import math
import json

import numpy as np
import pandas as pd

from .sample_metric import SampleMetric
from ..dataset_creator import DatasetCreator

class MassingContextualRelevance(SampleMetric):
    def __init__(self, *, massing_key: str,
                        context_key: str,
                        additional_features: List[str],
                        massing_metric: SampleMetric,
                        id_key: str,
                        context_dataset_loader: DatasetCreator,
                        context_reduction: Literal["min", "max", "mean", "nearest"]) -> None:
        self.massing_key = massing_key
        self.context_key = context_key
        self.additional_features = additional_features
        self.massing_metric = massing_metric
        self.id_key = id_key
        self.context_dataset_loader = context_dataset_loader
        self.context_reduction = context_reduction

        context_dataset = pd.DataFrame(self.context_dataset_loader())
        context_dataset["id_buffer"] = context_dataset["id"]
        context_dataset = context_dataset.set_index("id_buffer")
        self.context_dataset = context_dataset

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        massing = sample[self.massing_key]
        massing_sample = {
            "massing": massing,
            **{k: sample[k] for k in self.additional_features}
        }
        massing_value = self.massing_metric(sample=massing_sample)
        if np.isnan(massing_value):
            return 0

        context = self.context_dataset.loc[sample[self.context_key]]
        context_values = []
        for _, row in context.iterrows():
            context_sample = {
                "massing": row["massing"],
                **{k: row[k] for k in self.additional_features}
            }
            context_value = self.massing_metric(sample=context_sample)
            if not np.isnan(context_value):
                context_values.append(context_value)

        if self.context_reduction == "min":
            context_value = min(context_values)
        elif self.context_reduction == "max":
            context_value = max(context_values)
        elif self.context_reduction == "mean":
            context_value = np.mean(context_values)
        elif self.context_reduction == "nearest":
            context_value = context_values[np.argmin(np.abs(context_values - massing_value))]
        else:
            raise

        context_std = np.std(context_values)
        output = abs(context_value - massing_value) / context_std
        output = np.exp(-output)
        return output