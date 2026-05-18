from typing import List, Any, Dict, Literal
import base64
import json
from pathlib import Path

from .memory_to_dict_parser import MemoryToDictParser
from ..agent_message import AgentMessage, AgentMessageContent

class ImagesParser(MemoryToDictParser):
    def __init__(self, *, image_modality: str,
                        images_type: Literal["path", "url"]) -> None:
        self.image_modality = image_modality
        self.images_type = images_type

    def __call__(self, *, memory: List[AgentMessage]) -> Dict[str, Any]:
        messages = []
        for message in memory:
            message_dict = {}
            contents = []
            for content in message.content:
                if content.modality == self.image_modality:
                    if self.images_type == "path":
                        image_path = content.content
                        with open(image_path, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
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
                    elif self.images_type == "url":
                        content_dict = {
                            "type":"image_url",
                            "image_url":{
                                "url":content.content
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