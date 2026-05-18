from abc import ABC, abstractmethod
from typing import Any
import base64
from io import BytesIO

from PIL import Image

from .string_deserializer import StringDeserializer
from .image_bytes_deserializer import ImageBytesDeserializer

class ImageBytesSaver(StringDeserializer):
    def __init__(self, *, deserializer: ImageBytesDeserializer,
                        path: str) -> None:
        self.deserializer = deserializer
        self.path = path

    def __call__(self, *, string: str) -> Any:
        image = self.deserializer(string=string)
        image.save(self.path)
        return self.path