from typing import Any, List, Dict, Optional

from .base import Function
from .parsers import Parser

class FunctionWrapper(Function):
    def __init__(self, *, function: Function, input_parser: Optional[Parser] = None, output_parser: Optional[Parser] = None) -> None:
        self.function = function
        self.input_parser = input_parser
        self.output_parser = output_parser
    
    def __call__(self, **kwargs: Any) -> Any:
        params = {}
        if self.input_parser:
            args_params = self.input_parser(inputs=kwargs)
            params.update(args_params)
        else:
            params.update(kwargs)
        output = self.function(**params)
        if self.output_parser:
            output = self.output_parser(inputs=output)
        return output

class FunctionGraph(Function):
    def __init__(self, *, functions: List[Function], output_parser: Optional[Parser] = None) -> None:
        self.functions = functions
        self.output_parser = output_parser
    
    def __call__(self, **kwargs: Any) -> Any:
        state = {}
        state.update(kwargs)
        for func in self.functions:
            output = func(**state)
            if isinstance(output, dict):
                state.update(output)
        if self.output_parser:
            output = self.output_parser(inputs=state)
        else:
            output = state
        return output

class FunctionMap(Function):
    def __init__(self, *, function: Function, const_args: List[str], mapping_args: List[str]) -> None:
        self.function = function
        self.const_args = const_args
        self.mapping_args = mapping_args
        assert len(mapping_args) > 0, "At least one mapping argument is required"
    
    def __call__(self, **kwargs: Any) -> List[Any]:
        mapping_values = [kwargs[k] for k in self.mapping_args]
        assert len(set([len(v) for v in mapping_values])) == 1, "All mapping values must have the same length"
        args = {k:v for k, v in kwargs.items() if k in self.const_args}
        args.update({k:v for k, v in zip(self.mapping_args, mapping_values)})
        return [self.function(**{k:args[k][i] for k in args.keys()}) for i in range(len(mapping_values[0]))]

class FunctionLoop(Function):
    def __init__(self, *, function: Function, batch_size: int) -> None:
        self.function = function
        self.batch_size = batch_size
    
    def __call__(self, **kwargs: Dict[str, List[Any]]) -> List[Any]:
        keys = list(kwargs.keys())
        n_samples = len(kwargs[keys[0]])
        n_batches = n_samples // self.batch_size + 1 if n_samples % self.batch_size != 0 else n_samples // self.batch_size
        batches = [{k: [kwargs[k][j] for j in range(i*self.batch_size, max((i+1)*self.batch_size, n_samples))] for k in keys} for i in range(n_batches)]
        outputs = {k: [] for k in keys}
        for batch in batches:
            local_outputs = self.function(**batch)
            for k in keys:
                outputs[k].extend(local_outputs[k])
        return outputs

class FunctionInit(Function):
    def __new__(cls, *, function: Function, inputs: Dict[str, Any]) -> Any:
        return function(**inputs)
