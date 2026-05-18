from abc import ABC, abstractmethod
from typing import Any
import base64
from io import BytesIO

from PIL import Image

from .serializer import Serializer

class ImageUrlSerializer(Serializer):
    def __call__(self, *, obj: Any) -> str:
        buffer = BytesIO()
        obj.save(buffer, format='png')
        img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mime_type = "image/png"
        image_url = f"data:{mime_type};base64,{img_data}"
        return image_url