from typing import List, Any, Dict
import base64
import json
from pathlib import Path

from PIL import Image

from .memory_to_dict_parser import MemoryToDictParser
from ..agent_message import AgentMessage, AgentMessageContent
from ..serializer import ImageBytesSerializer

class ResizeImagesParser(MemoryToDictParser):
    def __init__(self, *, image_modality: str,
                        image_size: List[int]) -> None:
        self.image_modality = image_modality
        self.image_size = image_size
        self.serializer = ImageBytesSerializer()

    def __call__(self, *, memory: List[AgentMessage]) -> Dict[str, Any]:
        messages = []
        for message in memory:
            message_dict = {}
            contents = []
            for content in message.content:
                if content.modality == self.image_modality:
                    image_path = content.content
                    image = Image.open(image_path)
                    image = image.resize(self.image_size)
                    img_data = self.serializer(obj=image)
                    file_extension = Path(image_path).suffix.lower()
                    mime_type = f"image/{file_extension[1:]}" if file_extension else "image/jpeg"
                    image_url = f"data:{mime_type};base64,{img_data}"
                    content_dict = {
                        "type":"image_url",
                        "image_url":{
                            "url":image_url
                        }
                    }
                    contents.append(content_dict)
                else:
                    contents.append({})
            message_dict["role"] = message.role
            message_dict["content"] = contents
            messages.append(message_dict)
        output = {
            "messages":messages
        }
        return output