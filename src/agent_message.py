from dataclasses import dataclass
from typing import List

@dataclass
class AgentMessageContent:
    modality: str
    content: str

@dataclass
class AgentMessage:
    role: str
    content: List[AgentMessageContent]