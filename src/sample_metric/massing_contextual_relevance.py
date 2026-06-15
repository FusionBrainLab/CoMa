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

        # The context massings come from a fixed pool that is shared across all
        # samples, so the characteristic of a given context massing only depends
        # on its id. We precompute the characteristic once per unique context id
        # and cache it. This is numerically identical to recomputing it for every
        # sample, but avoids the O(num_samples * context_size) blow-up. After the
        # cache is built the (large) context dataframe is dropped so that many
        # metric instances can coexist in memory.
        context_dataset = pd.DataFrame(self.context_dataset_loader())
        ids = context_dataset[self.id_key].tolist()
        massings = context_dataset[self.massing_key].tolist()
        extra = {k: context_dataset[k].tolist() for k in self.additional_features}
        cache = {}
        for i in range(len(ids)):
            context_sample = {
                self.massing_key: massings[i],
                **{k: extra[k][i] for k in self.additional_features}
            }
            try:
                value = float(self.massing_metric(sample=context_sample))
                if not np.isfinite(value):
                    value = np.nan
            except Exception:
                value = np.nan
            cache[ids[i]] = value
        self.context_cache = cache
        del context_dataset

    def __call__(self, *, sample: Dict[str, Any]) -> float:
        massing = sample[self.massing_key]
        massing_sample = {
            self.massing_key: massing,
            **{k: sample[k] for k in self.additional_features}
        }
        massing_value = float(self.massing_metric(sample=massing_sample))
        if not np.isfinite(massing_value):
            raise ValueError("non-finite massing value")

        context_ids = sample[self.context_key]
        context_values = [self.context_cache[i] for i in context_ids if i in self.context_cache]
        context_values = np.asarray([v for v in context_values if np.isfinite(v)], dtype=float)
        if context_values.size == 0:
            raise ValueError("no valid context values")

        if self.context_reduction == "min":
            context_value = context_values.min()
        elif self.context_reduction == "max":
            context_value = context_values.max()
        elif self.context_reduction == "mean":
            context_value = context_values.mean()
        elif self.context_reduction == "nearest":
            context_value = context_values[np.argmin(np.abs(context_values - massing_value))]
        else:
            raise

        context_std = np.std(context_values)
        output = abs(context_value - massing_value) / context_std
        output = np.exp(-output)
        return output