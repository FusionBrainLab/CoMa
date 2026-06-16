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

class ContourHeatmapDatasetVisualizer(ImageDataVisualizer):
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
        ax = plt.gca()
        x_grid, y_grid = np.meshgrid(
            np.arange(len(heatmap_df.columns)) + 0.5,
            np.arange(len(heatmap_df.index)) + 0.5,
        )
        x_smooth = np.linspace(0, len(heatmap_df.columns), 300)
        y_smooth = np.linspace(0, len(heatmap_df.index), 300)
        z_by_x = np.array([np.interp(x_smooth, x_grid[0], row) for row in heatmap_df.values])
        z_smooth = np.array([np.interp(y_smooth, y_grid[:, 0], z_by_x[:, i]) for i in range(len(x_smooth))]).T
        levels = np.linspace(np.nanmin(z_smooth), np.nanmax(z_smooth), 10)
        contourf = ax.contourf(x_smooth, y_smooth, z_smooth, levels=levels, cmap=self.palette)
        for y_idx, y_value in enumerate(heatmap_df.index):
            for x_idx, x_value in enumerate(heatmap_df.columns):
                ax.text(x_idx + 0.5, y_idx + 0.5, f"{heatmap_df.loc[y_value, x_value]:.2f}", ha="center", va="center")
        ax.set_xlim(0, len(heatmap_df.columns))
        ax.set_ylim(len(heatmap_df.index), 0)
        ax.set_xticks(np.arange(len(heatmap_df.columns)) + 0.5)
        ax.set_yticks(np.arange(len(heatmap_df.index)) + 0.5)
        ax.set_xticklabels(heatmap_df.columns)
        ax.set_yticklabels(heatmap_df.index)
        ax.set_xticks(np.arange(len(heatmap_df.columns) + 1), minor=True)
        ax.set_yticks(np.arange(len(heatmap_df.index) + 1), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.35)
        ax.set_axisbelow(False)
        fig.colorbar(contourf, ax=ax).set_label(
            self.feature_renaming.get(self.value_feature, self.value_feature),
            fontsize=12,
        )
        plt.title(textwrap.fill(self.title, width=int(fig.get_figwidth() * 8)), fontsize=16, pad=20)
        plt.xlabel(self.feature_renaming.get(self.x_feature, self.x_feature), fontsize=12)
        plt.ylabel(self.feature_renaming.get(self.y_feature, self.y_feature), fontsize=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=600, bbox_inches='tight')
        buf.seek(0)

        image = Image.open(buf)
        return image