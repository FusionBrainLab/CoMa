from abc import ABC, abstractmethod
from typing import Any, Dict
import base64
from io import BytesIO
import io
import math

from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from .image_data_visualizer import ImageDataVisualizer

class SeabornBarplotCreator(ImageDataVisualizer):
    def __init__(self, *, dict_counter_key: str,
                        title_key: str,
                        x_label_key: str,
                        y_label_key: str) -> None:
        self.dict_counter_key = dict_counter_key
        self.title_key = title_key
        self.x_label_key = x_label_key
        self.y_label_key = y_label_key

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        value_counts = data[self.dict_counter_key]
        categories = [c for c in value_counts.keys()]
        values = [value_counts[c] for c in categories]
        df = pd.DataFrame({"category":categories, "value":values})
        df = df.sort_values(by="value")

        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x='category', y='value')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.title(data[self.title_key], fontsize=16, pad=20)
        plt.xlabel(data[self.x_label_key], fontsize=12)
        plt.ylabel(data[self.y_label_key], fontsize=12)

        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=600, bbox_inches='tight')
        buf.seek(0)

        image = Image.open(buf)
        return image