from typing import Any, Protocol, Dict, List, Optional
from dataclasses import dataclass
import json

from .logger import Logger
from .tqdm_logger import LogConfig

class JsonLogger(Logger):
    def __init__(self, *, log_configs: List[LogConfig],
                        log_path: str) -> None:
        self.log_configs = log_configs
        self.log_path = log_path
        self.logs = []
    
    def __call__(self, **kwargs: Any) -> None:
        for config in self.log_configs:
            if config.conditions_checker(**kwargs):
                self.logs.append(config.metrics_parser(**kwargs))
        with open(self.log_path, "w+") as f:
            json.dump(self.logs, f)