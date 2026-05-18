from abc import ABC, abstractmethod
from typing import Any, Protocol, Dict, List, Optional
from dataclasses import dataclass

from tqdm import tqdm

from .logger import Logger
from ..protocols import IndicatorProtocol, StringCreatorProtocol, DictCreatorProtocol

@dataclass
class LogConfig:
    conditions_checker: IndicatorProtocol
    metrics_parser: DictCreatorProtocol

class TqdmLogger(Logger):
    log_configs: List[LogConfig]
    init_condition_checker: IndicatorProtocol
    total_len_key: str
    desc_format_func: StringCreatorProtocol
    finish_condition_checker: IndicatorProtocol
    def __init__(self, *, log_configs: List[LogConfig],
                 init_condition_checker: IndicatorProtocol,
                 total_len_key: str,
                 step_key: str,
                 desc_format_func: StringCreatorProtocol,
                 finish_condition_checker: IndicatorProtocol) -> None:
        self.log_configs = log_configs
        self.init_condition_checker = init_condition_checker
        self.total_len_key = total_len_key
        self.step_key = step_key
        self.desc_format_func = desc_format_func
        self.finish_condition_checker = finish_condition_checker
        self.pbar = None
        self.step = 0
    
    def __call__(self, **kwargs: Any) -> None:
        if self.pbar is None and self.init_condition_checker(**kwargs):
            assert self.total_len_key in kwargs, f"Log kwargs must contain '{self.total_len_key}' at first call. "
            self.pbar = tqdm(total=kwargs[self.total_len_key])
        if self.pbar is not None:
            assert self.step_key in kwargs, f"Log kwargs must contain '{self.step_key}'. "
            cur_step = kwargs[self.step_key]
            update_value = max(cur_step - self.step, 0)
            for config in self.log_configs:
                if config.conditions_checker(**kwargs):
                    desc = self.desc_format_func(**kwargs)
                    self.pbar.update(update_value)
                    self.pbar.set_description(desc)
            self.step = cur_step
            
        if self.pbar is not None and self.finish_condition_checker(**kwargs):
            self.pbar.close()
            self.pbar = None