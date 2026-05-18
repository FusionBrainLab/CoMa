from typing import List, TypedDict, Dict, Any
import base64
from pathlib import Path
import io
from dataclasses import dataclass

import torch
from transformers import AutoProcessor
from PIL import Image

from .memory_tokenizer import MemoryTokenizer
from ..agent_message import AgentMessage
from ..core.attr_access import AttrSetter

@dataclass
class ParameterConfig:
    path: List[str|int]
    value: Any

class HFProcessor(MemoryTokenizer):
    def __init__(self, *, processor_name: str,
                        processor_kwargs: Dict[str, Any],
                        images_size: List[int] | None,
                        processor_parameter_updates: List[ParameterConfig]) -> None:
        self.processor_name = processor_name
        self.processor_kwargs = processor_kwargs
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.images_size = images_size
        self.processor_parameter_updates = processor_parameter_updates
        self.attr_setter = AttrSetter()

    def __call__(self, *, memories: List[List[AgentMessage]]) -> Dict[str, Any]:
        for upd in self.processor_parameter_updates:
            self.attr_setter(obj=self.processor, path=upd.path, value=upd.value)

        messages_batch = []
        for memory in memories:
            messages = []
            for message in memory:
                message_dict = {}
                contents = []
                for content in message.content:
                    if content.modality == "image":
                        image = Image.open(content.content)
                        if self.images_size != None:
                            image = image.resize(self.images_size)
                        buffered = io.BytesIO()
                        
                        image_path = content.content
                        file_extension = Path(image_path).suffix.lower()[1:]
                        image.save(buffered, format="PNG")

                        img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        mime_type = f"image/png"
                        image_url = f"data:{mime_type};base64,{img_data}"

                        content_dict = {
                            "type":"image",
                            "image":image
                        }
                        contents.append(content_dict)
                    elif content.modality == "text":
                        contents.append({
                            "type":"text",
                            "text":content.content
                        })
                message_dict["role"] = message.role
                message_dict["content"] = contents
                messages.append(message_dict)
            messages_batch.append(messages)
        
        inputs = self.processor.apply_chat_template(
            messages_batch,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **self.processor_kwargs
        )
        result = {k: inputs[k] for k in inputs.keys()}

        
        return result