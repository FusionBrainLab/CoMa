from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Protocol, Dict, Any

from .langgraph_condition_function import LanggraphConditionalFunction
from ..memory_indicator import MemoryIndicator
from ..dict_to_memory_parser import DictToMemoryParser

class MemoryCondition(LanggraphConditionalFunction):
    def __init__(self, *, memory_indicator: MemoryIndicator,
                        memory_parser: DictToMemoryParser,
                        true_false_conditions: List[str]) -> None:
        self.memory_indicator = memory_indicator
        self.memory_parser = memory_parser
        self.true_false_conditions = true_false_conditions

    def __call__(self, *, state: Dict[str, Any]) -> str:
        memory = self.memory_parser(dictionary=state)
        value = self.memory_indicator(memory=memory)
        result = self.true_false_conditions[0] if value == True else self.true_false_conditions[1]
        return result

