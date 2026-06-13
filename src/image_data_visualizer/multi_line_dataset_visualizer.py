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

class MultiLineDatasetVisualizer(ImageDataVisualizer):
    def __init__(self, *, dataset_key: str,
                        x_feature: str,
                        y_feature: str,
                        line_feature: str,
                        title: str,
                        feature_renaming: Dict[str, str],
                        fixed_features: Dict[str, Any],
                        horizontal_lines: Dict[str, Dict[str, Any]]) -> None:
        self.dataset_key = dataset_key
        self.x_feature = x_feature
        self.y_feature = y_feature
        self.line_feature = line_feature
        self.title = title
        self.feature_renaming = feature_renaming
        self.fixed_features = fixed_features
        self.horizontal_lines = horizontal_lines

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        dataset = data[self.dataset_key]
        df = pd.DataFrame(dataset)
        base_df = df
        for feature, value in self.fixed_features.items():
            df = df[df[feature] == value]
        df = df.groupby(
            [self.line_feature, self.x_feature],
            as_index=False,
        )[self.y_feature].mean()
        line_values = sorted(df[self.line_feature].unique())
        colors = sns.color_palette("husl", len(line_values))

        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        for line_value, color in zip(line_values, colors):
            line_df = df[df[self.line_feature] == line_value].sort_values(by=self.x_feature)
            plt.plot(
                line_df[self.x_feature],
                line_df[self.y_feature],
                marker="o",
                color=color,
                label=str(line_value),
            )

        ax = plt.gca()
        horizontal_colors = sns.color_palette("Set2", len(self.horizontal_lines))
        for (line_name, line_filters), color in zip(self.horizontal_lines.items(), horizontal_colors):
            line_df = base_df
            for feature, value in line_filters.items():
                line_df = line_df[line_df[feature] == value]
            line_y = line_df[self.y_feature].mean()
            ax.axhline(line_y, linestyle="--", color=color)
            ax.text(ax.get_xlim()[0], line_y, line_name, color=color, va="bottom")

        plt.title(self.title, fontsize=16, pad=20)
        plt.xlabel(self.feature_renaming.get(self.x_feature, self.x_feature), fontsize=12)
        plt.ylabel(self.feature_renaming.get(self.y_feature, self.y_feature), fontsize=12)
        plt.legend(title=self.feature_renaming.get(self.line_feature, self.line_feature))
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=600, bbox_inches='tight')
        buf.seek(0)

        image = Image.open(buf)
        return image