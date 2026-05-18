from typing import List, Dict, Any

from .core.base import Function

class vLLMInference(Function):
    hub_model_path: str
    temperature: float
    top_p: float
    max_length: int
    tensor_parallel_size: int
    def __init__(self, *, hub_model_path: str,
                 hub_tokenizer_path: str,
                 tensor_parallel_size: int,
                 sampling_params: Dict[str, Any]) -> None:
        from vllm import LLM, SamplingParams
        self.hub_model_path = hub_model_path
        self.hub_tokenizer_path = hub_tokenizer_path
        """self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_length = max_length"""
        #self.sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, min_p=self.min_p, max_tokens=self.max_length)
        self.sampling_params = SamplingParams(**sampling_params)
        self.tensor_parallel_size = tensor_parallel_size
        self.vllm_model = LLM(model=self.hub_model_path, tensor_parallel_size=self.tensor_parallel_size, tokenizer=hub_tokenizer_path)
    
    def __call__(self, *, inputs: List[Dict[str, Any]]) -> List[str]:
        outputs = self.vllm_model.generate(inputs, self.sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        #print(outputs)
        return outputs