from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import base64
from io import BytesIO
import io
import math
import textwrap

from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd

from .image_data_visualizer import ImageDataVisualizer

class ContourHeatmapGridDatasetVisualizer(ImageDataVisualizer):
    def __init__(self, *, dataset_key: str,
                        x_feature: str,
                        y_feature: str,
                        value_feature: str,
                        x_grid_feature: str,
                        y_grid_feature: str,
                        title: Optional[str],
                        feature_renaming: Dict[str, str],
                        palette: str,
                        fixed_features: Dict[str, Any]) -> None:
        self.dataset_key = dataset_key
        self.x_feature = x_feature
        self.y_feature = y_feature
        self.value_feature = value_feature
        self.x_grid_feature = x_grid_feature
        self.y_grid_feature = y_grid_feature
        self.title = title
        self.feature_renaming = feature_renaming
        self.palette = palette
        self.fixed_features = fixed_features

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        dataset = data[self.dataset_key]
        df = pd.DataFrame(dataset)
        for feature, value in self.fixed_features.items():
            df = df[df[feature] == value]
        x_grid_values = sorted(df[self.x_grid_feature].unique())
        y_grid_values = sorted(df[self.y_grid_feature].unique())
        heatmap_dfs = {}
        value_min = df[self.value_feature].min()
        value_max = df[self.value_feature].max()

        sns.set_style("whitegrid")
        for y_grid_value in y_grid_values:
            for x_grid_value in x_grid_values:
                cell_df = df[
                    (df[self.x_grid_feature] == x_grid_value)
                    & (df[self.y_grid_feature] == y_grid_value)
                ]
                heatmap_dfs[(y_grid_value, x_grid_value)] = cell_df.groupby(
                    [self.y_feature, self.x_feature],
                    as_index=False,
                )[self.value_feature].mean().pivot(
                    index=self.y_feature,
                    columns=self.x_feature,
                    values=self.value_feature,
                ).sort_index().sort_index(axis=1)

        fig, axes = plt.subplots(
            len(y_grid_values),
            len(x_grid_values),
            figsize=(6.5 * len(x_grid_values), 4 * len(y_grid_values)),
            squeeze=False,
        )
        norm = colors.Normalize(vmin=value_min, vmax=value_max)
        for row_idx, y_grid_value in enumerate(y_grid_values):
            for col_idx, x_grid_value in enumerate(x_grid_values):
                ax = axes[row_idx][col_idx]
                heatmap_df = heatmap_dfs[(y_grid_value, x_grid_value)]
                x_grid, y_grid = np.meshgrid(
                    np.arange(len(heatmap_df.columns)) + 0.5,
                    np.arange(len(heatmap_df.index)) + 0.5,
                )
                x_smooth = np.linspace(0, len(heatmap_df.columns), 300)
                y_smooth = np.linspace(0, len(heatmap_df.index), 300)
                z_by_x = np.array([np.interp(x_smooth, x_grid[0], row) for row in heatmap_df.values])
                z_smooth = np.array([np.interp(y_smooth, y_grid[:, 0], z_by_x[:, i]) for i in range(len(x_smooth))]).T
                levels = np.linspace(np.nanmin(z_smooth), np.nanmax(z_smooth), 10)
                ax.contourf(x_smooth, y_smooth, z_smooth, levels=levels, cmap=self.palette, norm=norm)
                for y_idx, y_value in enumerate(heatmap_df.index):
                    for x_idx, x_value in enumerate(heatmap_df.columns):
                        ax.text(x_idx + 0.5, y_idx + 0.5, f"{heatmap_df.loc[y_value, x_value]:.2f}", ha="center", va="center", fontsize=7)
                ax.set_xlim(0, len(heatmap_df.columns))
                ax.set_ylim(len(heatmap_df.index), 0)
                ax.set_xticks(np.arange(len(heatmap_df.columns)) + 0.5)
                ax.set_yticks(np.arange(len(heatmap_df.index)) + 0.5)
                ax.set_xticklabels(heatmap_df.columns)
                ax.set_yticklabels(heatmap_df.index)
                ax.set_xticks(np.arange(len(heatmap_df.columns) + 1), minor=True)
                ax.set_yticks(np.arange(len(heatmap_df.index) + 1), minor=True)
                ax.grid(False)
                ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.3)
                ax.set_axisbelow(False)
                ax.tick_params(axis="both", labelsize=8)
                ax.set_xlabel(self.feature_renaming.get(self.x_feature, self.x_feature), fontsize=9)
                ax.set_ylabel(self.feature_renaming.get(self.y_feature, self.y_feature), fontsize=9)
                if row_idx == 0:
                    ax.set_title(f"{self.feature_renaming.get(self.x_grid_feature, self.x_grid_feature)}={x_grid_value}", fontsize=12, pad=10)
                if col_idx == 0:
                    ax.text(
                        -0.18,
                        0.5,
                        f"{self.feature_renaming.get(self.y_grid_feature, self.y_grid_feature)}={y_grid_value}",
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=12,
                    )
        fig.subplots_adjust(top=0.9 if self.title is not None else 0.96, right=0.86, wspace=0.25, hspace=0.35)
        colorbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=self.palette), cax=colorbar_ax)
        colorbar.set_label(
            self.feature_renaming.get(self.value_feature, self.value_feature),
            fontsize=12,
        )
        if self.title is not None:
            fig.suptitle(textwrap.fill(self.title, width=int(fig.get_figwidth() * 8)), fontsize=16)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
        buf.seek(0)

        image = Image.open(buf)
        return image