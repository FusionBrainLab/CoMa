from typing import Any, Protocol, Dict, List, Optional
from dataclasses import dataclass

import wandb

from .logger import Logger
from ..protocols import IndicatorProtocol, DictCreatorProtocol
from .tqdm_logger import LogConfig

@dataclass
class WandbStepMetricConfig:
    step_metric: str
    related_metrics: List[str]

class WandbLogger(Logger):
    init_condition_checker: IndicatorProtocol
    project: str
    name: str
    notes: str
    tags: List[str]
    config_creator: DictCreatorProtocol
    log_configs: List[LogConfig]
    step_metrics_configs: List[WandbStepMetricConfig]
    summary_creator: DictCreatorProtocol
    finish_condition_checker: IndicatorProtocol
    def __init__(self, *, init_condition_checker: IndicatorProtocol,
                    project: str,
                    name: str,
                    id: str,
                    notes: str,
                    tags: List[str],
                    login: bool,
                    resume: Optional[str],
                    resume_from_key: Optional[str],
                    config_creator: DictCreatorProtocol,
                    log_configs: List[LogConfig],
                    step_metrics_configs: List[WandbStepMetricConfig],
                    summary_creator: DictCreatorProtocol,
                    finish_condition_checker: IndicatorProtocol) -> None:
        self.log_configs = log_configs
        self.init_condition_checker = init_condition_checker
        self.project = project
        self.name = name
        self.id = id
        self.notes = notes
        self.tags = tags
        self.login = login
        self.resume = resume
        self.resume_from_key = resume_from_key
        self.config_creator = config_creator
        self.step_metrics_configs = step_metrics_configs
        self.summary_creator = summary_creator
        self.finish_condition_checker = finish_condition_checker
        
        self.run = None
    
    def __call__(self, **kwargs: Any) -> None:
        if self.run == None and self.init_condition_checker(**kwargs):
            if self.login:
                wandb.login(relogin=True)
            resume = self.resume
            resume_from = None
            if self.resume_from_key != None:
                assert self.resume_from_key in kwargs
                resume_from = f"{self.id}?_step={str(kwargs[self.resume_from_key])}"
            self.run = wandb.init(project=self.project,
                                  name=self.name,
                                  id=self.id,
                                  notes=self.notes,
                                  tags=self.tags,
                                  config=self.config_creator(**kwargs),
                                  resume=resume,
                                  resume_from=resume_from)
            for step_metric_config in self.step_metrics_configs:
                for metric in step_metric_config.related_metrics:
                    self.run.define_metric(metric, step_metric=step_metric_config.step_metric)
        if self.run != None:
            for config in self.log_configs:
                if config.conditions_checker(**kwargs):
                    self.run.log(config.metrics_parser(**kwargs))
        if self.run != None and self.finish_condition_checker(**kwargs):
            self.run.summary = self.summary_creator(**kwargs)
            self.run.finish()
            self.run = None