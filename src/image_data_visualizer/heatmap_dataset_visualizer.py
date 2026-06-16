from abc import ABC, abstractmethod
from typing import Any, Dict
import base64
from io import BytesIO
import io
import math
import textwrap

from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from .image_data_visualizer import ImageDataVisualizer

class HeatmapDatasetVisualizer(ImageDataVisualizer):
    def __init__(self, *, dataset_key: str,
                        x_feature: str,
                        y_feature: str,
                        value_feature: str,
                        title: str,
                        feature_renaming: Dict[str, str],
                        palette: str,
                        fixed_features: Dict[str, Any]) -> None:
        self.dataset_key = dataset_key
        self.x_feature = x_feature
        self.y_feature = y_feature
        self.value_feature = value_feature
        self.title = title
        self.feature_renaming = feature_renaming
        self.palette = palette
        self.fixed_features = fixed_features

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        dataset = data[self.dataset_key]
        df = pd.DataFrame(dataset)
        for feature, value in self.fixed_features.items():
            df = df[df[feature] == value]
        heatmap_df = df.groupby(
            [self.y_feature, self.x_feature],
            as_index=False,
        )[self.value_feature].mean().pivot(
            index=self.y_feature,
            columns=self.x_feature,
            values=self.value_feature,
        ).sort_index().sort_index(axis=1)

        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(10, 6))
        ax = sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap=self.palette)
        plt.title(textwrap.fill(self.title, width=int(fig.get_figwidth() * 8)), fontsize=16, pad=20)
        plt.xlabel(self.feature_renaming.get(self.x_feature, self.x_feature), fontsize=12)
        plt.ylabel(self.feature_renaming.get(self.y_feature, self.y_feature), fontsize=12)
        ax.collections[0].colorbar.set_label(
            self.feature_renaming.get(self.value_feature, self.value_feature),
            fontsize=12,
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=600, bbox_inches='tight')
        buf.seek(0)

        image = Image.open(buf)
        return image