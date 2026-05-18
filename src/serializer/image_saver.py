from abc import ABC, abstractmethod
from typing import Any
import base64
from io import BytesIO

from PIL import Image

from .serializer import Serializer
from ..string_formatter import StringFormatter

class ImageSaver(Serializer):
    def __init__(self, *, path_template: str,
                        image_key: str) -> None:
        self.path_template = path_template
        self.image_key = image_key
        self.string_formatter = StringFormatter()

    def __call__(self, *, obj: Any) -> str:
        image = obj[self.image_key]
        path = self.string_formatter(string=self.path_template, args=obj)
        image.save(path)
        return path