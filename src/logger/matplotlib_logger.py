from typing import Any, Protocol, Dict, List, Optional
from dataclasses import dataclass
import os

import matplotlib.pyplot as plt

from .logger import Logger
from ..protocols import IndicatorProtocol, DictCreatorProtocol
from .tqdm_logger import LogConfig

class MatplotlibLogger(Logger):
    def __init__(self, *, log_configs: List[LogConfig],
                        log_folder: str) -> None:
        self.log_configs = log_configs
        self.log_folder = log_folder
        self.metrics = {}
    
    def __call__(self, **kwargs: Any) -> None:
        update_metrics = []
        for config in self.log_configs:
            if config.conditions_checker(**kwargs):
                local_metrics = config.metrics_parser(**kwargs)
                for k, v in local_metrics.items():
                    update_metrics.append(k)
                    if k not in self.metrics:
                        self.metrics[k] = []
                    self.metrics[k].append(v)
        for metric in update_metrics:
            values = self.metrics[metric]
            x = list(range(len(values)))
            plt.plot(x, values)
            plt.xlabel('step')
            plt.ylabel(metric)
            path = os.path.join(self.log_folder, f"{metric}.png")
            plt.savefig(path)