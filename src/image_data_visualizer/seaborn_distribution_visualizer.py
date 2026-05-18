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

class SeabornDistributionVisualizer(ImageDataVisualizer):
    def __init__(self, *, values_key: str,
                        fontsize: int,
                        palette: str,
                        log_scale: bool,
                        kde: bool) -> None:
        self.values_key = values_key
        self.fontsize = fontsize
        self.palette = palette
        self.log_scale = log_scale
        self.kde = kde

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        plot_df = pd.DataFrame({"values":data[self.values_key]})

        sns.set_style("whitegrid") 
        plt.figure(figsize=(10, 6))

        args = {
            "data":plot_df, "x":"values",
            "fill":True, "common_norm":False, "palette":self.palette,
            "alpha":0.2, "height":6, "aspect":1.5, "log_scale":self.log_scale
        }
        if self.kde:
            args["kind"] = "kde"
        ax = sns.displot(**args)

        plt.xlabel('Values', fontsize=self.fontsize)
        plt.ylabel('Density', fontsize=self.fontsize)
        ax.tick_params(axis='x', labelsize=self.fontsize)
        ax.tick_params(axis='y', labelsize=self.fontsize)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=600, bbox_inches='tight')
        buf.seek(0)

        image = Image.open(buf)
        return image