from typing import List, Any, Dict

from .dict_to_memory_parser import DictToMemoryParser
from ..core.attr_access import AttrGetter
from ..agent_message import AgentMessage, AgentMessageContent
        
class MessagesDTMParser(DictToMemoryParser):
    def __init__(self, *, messages_key: str) -> None:
        self.messages_key = messages_key

    def __call__(self, *, dictionary: Dict[str, Any]) -> List[AgentMessage]:
        messages = dictionary[self.messages_key]
        memory = []
        for m in messages:
            if type(m["content"]) == str:
                message = AgentMessage(role=m["role"], content=[AgentMessageContent(modality="text", content=m["content"])])
            else:
                contents = []
                for c in m["content"]:
                    content = AgentMessageContent(modality=c["type"], content=c["text"])
                    contents.append(content)
                message = AgentMessage(role=m["role"], content=contents)
            memory.append(message)
        return memory