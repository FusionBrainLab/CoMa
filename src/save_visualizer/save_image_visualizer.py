from typing import List, Dict, Any
import re
import json
import os

from PIL import Image
import pandas as pd
import numpy as np

from .save_visualizer import SaveVisualizer
from ..image_data_visualizer import ImageDataVisualizer

class SaveImageVisualizer(SaveVisualizer):
    def __init__(self, *, image_visualizer: ImageDataVisualizer,
                        path: str) -> None:
        self.image_visualizer = image_visualizer
        self.path = path
    def __call__(self, *, data: Dict[str, Any]) -> str:
        image = self.image_visualizer(data=data)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        image.save(self.path)
        return self.path