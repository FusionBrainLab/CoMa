from abc import ABC, abstractmethod
from typing import Any
import base64
from io import BytesIO

from PIL import Image

from .string_deserializer import StringDeserializer

class ImageBytesDeserializer(StringDeserializer):
    def __call__(self, *, string: str) -> Any:
        image_bytes = base64.b64decode(string)
        image = Image.open(BytesIO(image_bytes))
        return image