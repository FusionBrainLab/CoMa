from abc import ABC, abstractmethod
from typing import Any
import base64
from io import BytesIO

from PIL import Image

from .serializer import Serializer

class ImageBytesSerializer(Serializer):
    def __call__(self, *, obj: Any) -> str:
        image = obj
        buffered = BytesIO()
        image.save(buffered, format="png")
        img_bytes = buffered.getvalue()
        img_b64_bytes = base64.b64encode(img_bytes)
        img_b64_string = img_b64_bytes.decode('utf-8')
        return img_b64_string