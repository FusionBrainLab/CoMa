from typing import List, Any, Dict

from .dict_to_memory_parser import DictToMemoryParser
from ..agent_message import AgentMessage
        
class EmptyColAdaptiveDTMParser(DictToMemoryParser):
    def __init__(self, *, combination_to_cols: Dict[str, List[str]],
                        combination_to_parser: Dict[str, DictToMemoryParser],
                        empty_parser: DictToMemoryParser) -> None:
        self.combination_to_cols = combination_to_cols
        self.combination_to_parser = combination_to_parser
        self.empty_parser = empty_parser

    def __call__(self, *, dictionary: Dict[str, Any]) -> List[AgentMessage]:
        combination_cols = set()
        for cols in self.combination_to_cols.values():
            combination_cols.update(cols)

        all_empty = True
        for col in combination_cols:
            value = dictionary.get(col)
            is_empty = value is None or value != value or value == [] or value == {} or value == "" or value == "null"
            if not is_empty:
                all_empty = False
                break

        if all_empty:
            return self.empty_parser(dictionary=dictionary)

        for combination_name, cols in self.combination_to_cols.items():
            if combination_name not in self.combination_to_parser:
                continue

            required_cols = set(cols)
            is_match = True
            for col in combination_cols:
                value = dictionary.get(col)
                is_empty = value is None or value != value or value == [] or value == {} or value == "" or value == "null"
                if col in required_cols and is_empty:
                    is_match = False
                    break
                if col not in required_cols and not is_empty:
                    is_match = False
                    break

            if is_match:
                return self.combination_to_parser[combination_name](dictionary=dictionary)

        raise ValueError("No parser found for non-empty dictionary columns combination")