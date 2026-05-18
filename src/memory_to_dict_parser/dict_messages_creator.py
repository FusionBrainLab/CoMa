from typing import List, Any, Dict
import json

from .memory_to_dict_parser import MemoryToDictParser
from ..agent_message import AgentMessage, AgentMessageContent

class DictMessagesCreator(MemoryToDictParser):
    def __init__(self, *, flatten: bool) -> None:
        self.flatten = flatten

    def __call__(self, *, memory: List[AgentMessage]) -> Dict[str, Any]:
        messages = []
        for message in memory:
            message_dict = {}
            text_contents = []
            for content in message.content:
                text_contents.append(content.content)
            message_dict["role"] = message.role
            if self.flatten:
                content = "".join(text_contents)
                message_dict["content"] = content
            else:
                message_dict["content"] = []
                for content in text_contents:
                    if len(content) > 0:
                        message_dict["content"].append({"type":"text", "text":content})
                    else:
                        message_dict["content"].append({})
            messages.append(message_dict)
        output = {
            "messages":messages
        }
        return output