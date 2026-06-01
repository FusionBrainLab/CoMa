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

class ModalityFilterTokenizer(MemoryTokenizer):
    def __init__(self, *, base_tokenizer: MemoryTokenizer,
                        allowed_modalities: List[str]) -> None:
        self.base_tokenizer = base_tokenizer
        self.allowed_modalities = allowed_modalities

    def __call__(self, *, memories: List[List[AgentMessage]]) -> Dict[str, Any]:
        filtered_memories = []
        for memory in memories:
            new_memory = []
            for message in memory:
                new_content = []
                for content in message.content:
                    if content.modality in self.allowed_modalities:
                        new_content.append(content)
                new_message = AgentMessage(role=message.role, content=new_content)
                new_memory.append(new_message)
            filtered_memories.append(new_memory)
        return self.base_tokenizer(memories=filtered_memories)