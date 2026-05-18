from typing import List, Any, Dict

from .dict_to_memory_parser import DictToMemoryParser
from ..agent_message import AgentMessage, AgentMessageContent
from ..dict_to_content_parser import DictToContentParser
        
class ContentChainDTMParser(DictToMemoryParser):
    def __init__(self, *, role: str,
                        content_makers: List[DictToContentParser]) -> None:
        self.role = role
        self.content_makers = content_makers

    def __call__(self, *, dictionary: Dict[str, Any]) -> List[AgentMessage]:
        contents = []
        for parser in self.content_makers:
            content = parser(dictionary=dictionary)
            contents.append(content)
        memory = [AgentMessage(role=self.role, content=contents)]
        return memory