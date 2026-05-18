from typing import List, Any, Dict

from .metric import Metric
from ..agent import Agent
from ..dict_to_memory_parser import DictToMemoryParser
from ..memory_indicator import MemoryIndicator

class AgentIndicatorAccuracy(Metric):
    def __init__(self, *, agent: Agent,
                        memory_creator: DictToMemoryParser,
                        output_indicator: MemoryIndicator) -> None:
        self.agent = agent
        self.memory_creator = memory_creator
        self.output_indicator = output_indicator
        
    def __call__(self, *, submit: Dict[str, List[Any]]) -> float:
        keys = list(submit.keys())
        length = len(submit[keys[0]])
        memories = [self.memory_creator(dictionary={k: submit[k][i] for k in keys}) for i in range(length)]
        output_memories = self.agent(memories=memories)
        indicators = [self.output_indicator(memory=output_memory) for output_memory in output_memories]
        true_counter = 0
        for ind in indicators:
            if ind == True:
                true_counter += 1
        accuracy = true_counter / length
        return accuracy