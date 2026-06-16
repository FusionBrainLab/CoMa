from typing import List, Dict, Any

from .core.base import Function

class vLLMInference(Function):
    def __init__(self, *, init_params: Dict[str, Any],
                 sampling_params: Dict[str, Any]) -> None:
        from vllm import LLM, SamplingParams
        self.init_params = init_params
        self.sampling_params = SamplingParams(**sampling_params)
        self.vllm_model = LLM(**init_params)
    
    def __call__(self, *, inputs: List[Dict[str, Any]]) -> List[str]:
        outputs = self.vllm_model.generate(inputs, self.sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs