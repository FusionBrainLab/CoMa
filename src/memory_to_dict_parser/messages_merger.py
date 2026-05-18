from typing import List, Any, Dict

from .memory_to_dict_parser import MemoryToDictParser
from ..agent_message import AgentMessage, AgentMessageContent

class MessagesMerger(MemoryToDictParser):
    def __init__(self, *, base_parsers: List[MemoryToDictParser],
                        messages_key: str,
                        flatten_content: bool) -> None:
        self.base_parsers = base_parsers
        self.messages_key = messages_key
        self.flatten_content = flatten_content

    def __call__(self, *, memory: List[AgentMessage]) -> Dict[str, Any]:
        base_outputs = [parser(memory=memory) for parser in self.base_parsers]
        base_messages = [o[self.messages_key] for o in base_outputs]
        output = {}
        new_messages = []
        for i in range(len(memory)):
            final_message = {}
            local_messages = [m[i] for m in base_messages]
            for m in local_messages:
                for key in m.keys():
                    if key != "content":
                        final_message[key] = m[key]
            if self.flatten_content:
                for m in local_messages:
                    if "content" in m:
                        final_message["content"] = m["content"]
                        break
            else:
                final_content = []
                for m in local_messages:
                    if "content" in m:
                        for i in range(len(m["content"])):
                            if len(final_content) <= i:
                                final_content.append(m["content"][i])
                            else:
                                if len(list(m["content"][i].keys())) > 0:
                                    final_content[i] = m["content"][i]
                final_message["content"] = final_content
            new_messages.append(final_message)
        output[self.messages_key] = new_messages
        return output
