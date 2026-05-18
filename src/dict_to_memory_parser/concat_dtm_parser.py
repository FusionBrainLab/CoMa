from typing import List, Any, Dict

from .dict_to_memory_parser import DictToMemoryParser
from ..core.attr_access import AttrGetter
from ..agent_message import AgentMessage, AgentMessageContent
        
class ConcatDTMParser(DictToMemoryParser):
    def __init__(self, *, base_parsers: List[DictToMemoryParser]) -> None:
        self.base_parsers = base_parsers

    def __call__(self, *, dictionary: Dict[str, Any]) -> List[AgentMessage]:
        memory = []
        for parser in self.base_parsers:
            memory.extend(parser(dictionary=dictionary))
        return memory