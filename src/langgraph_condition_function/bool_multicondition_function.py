from typing import List, Literal, Dict, Any

from .langgraph_condition_function import LanggraphConditionalFunction

class BoolMulticonditionFunction(LanggraphConditionalFunction):
    def __init__(self, *, condition_functions: List[LanggraphConditionalFunction],
                        bool_mapping: Dict[str, bool],
                        composer: Literal["and", "or"],
                        true_false_conditions: List[str]) -> None:
        self.condition_functions = condition_functions
        self.bool_mapping = bool_mapping
        self.composer = composer
        self.true_false_conditions = true_false_conditions

    def __call__(self, *, state: Dict[str, Any]) -> str:
        str_results = [f(state=state) for f in self.condition_functions]
        bool_results = [self.bool_mapping[r] for r in str_results]
        if self.composer == "and":
            result = all(bool_results)
        elif self.composer == "or":
            result = any(bool_results)
        output = self.true_false_conditions[0] if result == True else self.true_false_conditions[1]
        return output
