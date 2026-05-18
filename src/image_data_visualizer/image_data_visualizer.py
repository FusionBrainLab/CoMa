from abc import ABC, abstractmethod
from typing import List, Any, Dict

from PIL import Image

from ..core.base import Function

class ImageDataVisualizer(Function, ABC):
    @abstractmethod
    def __call__(self, *, data: Dict[str, Any]) -> Image:
        ...