from abc import ABC, abstractmethod
from typing import List, Any, Dict

from .dict_to_content_parser import DictToContentParser
from ..agent_message import AgentMessage, AgentMessageContent

class ModalityKeyDTCParser(DictToContentParser):
    def __init__(self, *, modality: str,
                        key: str) -> None:
        self.modality = modality
        self.key = key

    def __call__(self, *, dictionary: Dict[str, Any]) -> AgentMessageContent:
        return AgentMessageContent(modality=self.modality, content=dictionary[self.key])