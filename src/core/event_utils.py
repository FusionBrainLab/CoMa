from abc import ABC, abstractmethod
from typing import List, Any, Dict, Callable, Union, TypeVar, ParamSpec, Optional, Protocol
from dataclasses import dataclass
from functools import wraps

from .base import Function, Container
from .inheritance_utils import Descriptor, Decorator, MethodDecorationConfig, DynamicSubclassDecorator, AttributeDescriptionConfig
from .meta import Meta
from .attr_access import AttrGetter

class EventHandler(Protocol):
    def __call__(self, *, sender: object, event_name: str, event_args: Dict[str, Any]) -> None:
        ...

@dataclass
class EventConfig:
    event_name: str
    raiser_path: List[str]
    raise_before: bool
    raise_after: bool

class EventDecorator(Decorator):
    def __init__(self, pre_event: Optional[str], 
                 post_event: Optional[str],
                 sender: object,
                 handlers_storage: Dict[str, List[EventHandler]]) -> None:
        self.pre_event = pre_event
        self.post_event = post_event
        self.sender = sender
        self.handlers_storage = handlers_storage
    
    def __call__(self, obj: Any) -> Any:
        P = ParamSpec("P")
        R = TypeVar("R")
        @wraps(obj)
        def decorated_func(*args: P.args, **kwargs: P.kwargs) -> R:
            if self.pre_event is not None:
                event_args = {"INPUT": kwargs}
                handlers = self.handlers_storage[self.pre_event]
                for handler in handlers:
                    handler(sender=self.sender, event_name=self.pre_event, event_args=event_args)
            output = obj(*args, **kwargs)
            if self.post_event is not None:
                event_args = {"INPUT": kwargs, "OUTPUT": output}
                handlers = self.handlers_storage[self.post_event]
                for handler in handlers:
                    handler(sender=self.sender, event_name=self.post_event, event_args=event_args)
            return output
        return decorated_func

class EventHost(Function):
    def __init__(self, *, base_module: object,
                    event_configs: List[EventConfig],
                    sender: object,
                    handlers_storage: Dict[str, List[Any]]) -> None:
        self.base_module = base_module
        self.event_configs = event_configs
        self.sender = sender
        self.handlers_storage = handlers_storage
        self.local_raisers = {}
        local_configs = {}
        for config in event_configs:
            if len(config.raiser_path) == 1:
                local_raiser = config.raiser_path[0]
                if local_raiser not in self.local_raisers:
                    self.local_raisers[local_raiser] = []
                self.local_raisers[local_raiser].append(config)
            else:
                subraiser_name = config.raiser_path[0]
                if subraiser_name not in local_configs:
                    local_configs[subraiser_name] = []
                local_config = EventConfig(event_name=config.event_name,
                                           raiser_path=config.raiser_path[1:],
                                           raise_before=config.raise_before,
                                           raise_after=config.raise_after)
                local_configs[subraiser_name].append(local_config)
        for name, configs in local_configs.items():
            subraiser = getattr(self.base_module, name)
            wrapped_subraiser = EventHost(base_module=subraiser,
                                          event_configs=configs,
                                          sender=self.sender,
                                          handlers_storage=self.handlers_storage)
            setattr(self.base_module, name, wrapped_subraiser)
    
    def _get_method_decorator(self, *, method_name: str) -> EventDecorator:
        pre_event = None
        post_event = None
        for config in self.local_raisers[method_name]:
            if config.raise_before:
                pre_event = config.event_name
            if config.raise_after:
                post_event = config.event_name
        decorator = EventDecorator(pre_event=pre_event,
                                    post_event=post_event,
                                    sender=self.sender,
                                    handlers_storage=self.handlers_storage)
        return decorator
    
    def __call__(self, **kwargs: Any) -> Any:
        if not callable(self.base_module):
            raise AttributeError()
        if "__call__" in self.local_raisers:
            decorator = self._get_method_decorator(method_name="__call__")
            decorated_call = decorator(self.base_module.__call__)
            return decorated_call(**kwargs)
        else:
            return self.base_module(**kwargs)
    
    def __getattr__(self, name: str) -> Any:
        if name in self.local_raisers:
            decorator = self._get_method_decorator(method_name=name)
            method = getattr(self.base_module, name)
            decorated_method = decorator(obj=method)
            return decorated_method
        else:
            return getattr(self.base_module, name)

class EventableModule(EventHost):
    def __init__(self, *, base_module: object,
                    event_configs: List[EventConfig]) -> None:
        handlers_storage = {}
        for config in event_configs:
            handlers_storage[config.event_name] = []
        super().__init__(base_module=base_module,
                         event_configs=event_configs,
                         sender=base_module,
                         handlers_storage=handlers_storage)
        
    def subscribe(self, *, event_name: str, handler: EventHandler) -> None:
        self.handlers_storage[event_name].append(handler)
    
    def unsubscribe(self, *, event_name: str, handler: EventHandler) -> None:
        self.handlers_storage[event_name].remove(handler)

@dataclass
class EventHandlingStrategy:
    event_handler: EventHandler
    events_path: List[List[Union[str, int]]]

class EventHandlerRegistrator():
    def __new__(cls, eventable: EventableModule, event_handling_strategies: List[EventHandlingStrategy]):
        for strategy in event_handling_strategies:
            cls.apply_event_handling_strategy(eventable, strategy)
        return eventable

    @classmethod
    def apply_event_handling_strategy(cls, eventable: EventableModule, strategy: EventHandlingStrategy):
        for event_path in strategy.events_path:
            child_path = event_path[:-1]
            event = event_path[-1]
            if len(child_path) == 0:
                obj = eventable
            else:
                attr_getter = AttrGetter()
                obj = attr_getter(eventable, child_path)
            obj.subscribe(event_name=event, handler=strategy.event_handler)  