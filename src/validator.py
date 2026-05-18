from typing import List, Any, Dict
import json
from dataclasses import dataclass
import os

from .core.base import Function
from .dataset_creator import DatasetCreator
from .metric import Metric

@dataclass
class EvalConfig:
    method_name: str
    metric_name: str

class Validator(Function):
    def __init__(self, *, submit_loaders: Dict[str, DatasetCreator],
                        metrics: Dict[str, Metric],
                        eval_configs: List[EvalConfig],
                        result_storage_path: str) -> None:
        self.submit_loaders = submit_loaders
        self.metrics = metrics
        self.eval_configs = eval_configs
        self.result_storage_path = result_storage_path

    def __call__(self) -> None:
        result = {}
        methods = list(set(config.method_name for config in self.eval_configs))
        submits = {k: self.submit_loaders[k]() for k in methods}
        for config in self.eval_configs:
            submit = submits[config.method_name]
            metric = self.metrics[config.metric_name]
            local_result = metric(submit=submit)
            if config.method_name not in result:
                result[config.method_name] = {}
            result[config.method_name][config.metric_name] = local_result
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
