from abc import ABC, abstractmethod
from typing import List, Any, Dict

from .dict_to_content_parser import DictToContentParser
from ..agent_message import AgentMessage, AgentMessageContent
from ..string_formatter import StringFormatter

class StringFormatDTCParser(DictToContentParser):
    def __init__(self, *, template: str,
                        args_mapping: Dict[str, Any],
                        modality: str) -> None:
        self.template = template
        self.args_mapping = args_mapping
        self.modality = modality
        self.string_formatter = StringFormatter()

    def __call__(self, *, dictionary: Dict[str, Any]) -> AgentMessageContent:
        args = {k: dictionary[self.args_mapping[k]] for k in self.args_mapping.keys()}
        content = self.string_formatter(string=self.template, args=args)
        return AgentMessageContent(modality=self.modality, content=content)