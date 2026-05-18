from typing import List, Any, Dict
import json
import base64
from pathlib import Path
import io

from transformers import AutoProcessor
from PIL import Image

from .memory_to_dict_parser import MemoryToDictParser
from ..agent_message import AgentMessage, AgentMessageContent
from ..memory_tokenizer import ParameterConfig

class ProcessorVLLMInputCreator(MemoryToDictParser):
    def __init__(self, *, processor_name: str, 
                        processor_kwargs: Dict[str, Any],
                        images_size: List[int] | None,
                        processor_parameter_updates: List[ParameterConfig]) -> None:
        self.processor_name = processor_name
        self.processor_kwargs = processor_kwargs
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.images_size = images_size
        self.processor_parameter_updates = processor_parameter_updates

    def __call__(self, *, memory: List[AgentMessage]) -> Dict[str, Any]:
        messages = []
        images = []
        for message in memory:
            message_dict = {}
            contents = []
            for content in message.content:
                if content.modality == "image":
                    image = Image.open(content.content)
                    if self.images_size != None:
                        image = image.resize(self.images_size)
                    images.append(image)
                    buffered = io.BytesIO()
                    
                    image_path = content.content
                    file_extension = Path(image_path).suffix.lower()[1:]
                    image.save(buffered, format="PNG")

                    img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    mime_type = f"image/png"
                    image_url = f"data:{mime_type};base64,{img_data}"

                    content_dict = {
                        "type":"image",
                        "image":image_url
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

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            return_dict=True,
            **self.processor_kwargs
        )
        result = {
            "prompt":inputs,
            "multi_modal_data":{
                "image":images
            }
        }
        return result