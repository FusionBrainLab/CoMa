from typing import List, Any, Dict
import json
from dataclasses import dataclass
import os

from tqdm import tqdm

from .core.base import Function
from .dataset_creator import DatasetCreator
from .metric import Metric

class Validator(Function):
    def __init__(self, *, submit_loaders: Dict[str, DatasetCreator],
                        metrics: Dict[str, Metric],
                        submits_to_metrics: Dict[str, List[str]],
                        result_storage_path: str) -> None:
        self.submit_loaders = submit_loaders
        self.metrics = metrics
        self.submits_to_metrics = submits_to_metrics
        self.result_storage_path = result_storage_path

    def __call__(self) -> None:
        result = {}
        methods = list(set(self.submits_to_metrics.keys()))
        submits = {k: self.submit_loaders[k]() for k in methods}
        for method_name in methods:
            submit = submits[method_name]
            for metric_name in tqdm(self.submits_to_metrics[method_name], desc=f"Evaluating {method_name}"):
                metric = self.metrics[metric_name]
                local_result = metric(submit=submit)
                if method_name not in result:
                    result[method_name] = {}
                result[method_name][metric_name] = local_result
        if os.path.exists(self.result_storage_path):
            with open(self.result_storage_path, "r") as f:
                local_result = json.load(f)
            for method_name in local_result.keys():
                if method_name not in result:
                    result[method_name] = local_result[method_name]
                else:
                    for metric_name in local_result[method_name]:
                        if metric_name not in result[method_name].keys():
                            result[method_name][metric_name] = local_result[method_name][metric_name]
        with open(self.result_storage_path, "w+") as f:
            json.dump(result, f, ensure_ascii=False)
