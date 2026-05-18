from typing import List

from .memory_to_string_parser import MemoryToStringParser
from ..agent_message import AgentMessage

class ContentJoinMTSParser(MemoryToStringParser):
    def __init__(self, *, message_ind: int,
                        target_modalities: List[str],
                        join_str: str) -> None:
        self.message_ind = message_ind
        self.target_modalities = target_modalities
        self.join_str = join_str

    def __call__(self, *, memory: List[AgentMessage]) -> str:
        message = memory[self.message_ind]
        target_contents = []
        for c in message.content:
            if c.modality in self.target_modalities:
                target_contents.append(c.content)
        output = self.join_str.join(target_contents)
        return output
        