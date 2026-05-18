from typing import List, Any, Dict
import json

from .memory_to_dict_parser import MemoryToDictParser
from ..agent_message import AgentMessage, AgentMessageContent

class TextMessagesParser(MemoryToDictParser):
    def __init__(self, *, text_modalities: List[str],
                        flatten: bool) -> None:
        self.text_modalities = text_modalities
        self.flatten = flatten

    def __call__(self, *, memory: List[AgentMessage]) -> Dict[str, Any]:
        messages = []
        for message in memory:
            message_dict = {}
            text_contents = []
            for content in message.content:
                if content.modality in self.text_modalities:
                    text_contents.append(content.content)
                else:
                    text_contents.append("")
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