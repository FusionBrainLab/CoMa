from typing import List, Any, Dict

from .dict_to_memory_parser import DictToMemoryParser
from ..agent_message import AgentMessage
from ..dict_to_content_parser import DictToContentParser
        
class MultiColContentChainDTMParser(DictToMemoryParser):
    def __init__(self, *, role: str,
                        content_makers: Dict[str, DictToContentParser],
                        content_order: List[str],
                        content_key_to_col: Dict[str, str]) -> None:
        self.role = role
        self.content_makers = content_makers
        self.content_order = content_order
        self.content_key_to_col = content_key_to_col

    def __call__(self, *, dictionary: Dict[str, Any]) -> List[AgentMessage]:
        contents = []
        for content_key in self.content_order:
            parser = self.content_makers[content_key]
            col = self.content_key_to_col.get(content_key, None)
            if col != None:
                value = dictionary[col]
                is_empty = value is None or value != value or value == [] or value == {} or value == "" or value == "null"
                if is_empty:
                    continue
                if isinstance(value, list):
                    for item in value:
                        item_dictionary = dict(dictionary)
                        item_dictionary[col] = item
                        content = parser(dictionary=item_dictionary)
                        contents.append(content)
                    continue

            content = parser(dictionary=dictionary)
            contents.append(content)

        memory = [AgentMessage(role=self.role, content=contents)]
        return memory