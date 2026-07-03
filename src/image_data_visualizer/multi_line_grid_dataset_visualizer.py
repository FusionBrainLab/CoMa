from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
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

class MultiLineGridDatasetVisualizer(ImageDataVisualizer):
    def __init__(self, *, dataset_key: str,
                        x_feature: str,
                        y_feature: str,
                        line_features: List[str],
                        x_grid_feature: str,
                        y_grid_feature: str,
                        title: Optional[str],
                        feature_renaming: Dict[str, str],
                        legend_renaming: Dict[str, Dict[Any, str]],
                        fixed_features: Dict[str, Any],
                        horizontal_lines: Dict[str, Dict[str, Any]],
                        include_horizontal_line_samples: bool,
                        show_confidence_interval: bool,
                        palette: str,
                        horizontal_lines_palette: str) -> None:
        self.dataset_key = dataset_key
        self.x_feature = x_feature
        self.y_feature = y_feature
        self.line_features = line_features
        self.x_grid_feature = x_grid_feature
        self.y_grid_feature = y_grid_feature
        self.title = title
        self.feature_renaming = feature_renaming
        self.legend_renaming = legend_renaming
        self.fixed_features = fixed_features
        self.horizontal_lines = horizontal_lines
        self.include_horizontal_line_samples = include_horizontal_line_samples
        self.show_confidence_interval = show_confidence_interval
        self.palette = palette
        self.horizontal_lines_palette = horizontal_lines_palette

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        dataset = data[self.dataset_key]
        df = pd.DataFrame(dataset)
        base_df = df
        for feature, value in self.fixed_features.items():
            df = df[df[feature] == value]
        if not self.include_horizontal_line_samples:
            for line_filters in self.horizontal_lines.values():
                mask = pd.Series(True, index=df.index)
                for feature, value in line_filters.items():
                    mask = mask & (df[feature] == value)
                df = df[~mask]
        x_grid_values = sorted(df[self.x_grid_feature].unique())
        y_grid_values = sorted(df[self.y_grid_feature].unique())

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(
            len(y_grid_values),
            len(x_grid_values),
            figsize=(6.5 * len(x_grid_values), 4 * len(y_grid_values)),
            squeeze=False,
        )
        line_styles = ["-", "--", "-.", ":"]
        markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]
        if len(self.line_features) == 0:
            raise ValueError("MultiLineDatasetVisualizer requires at least one line feature")
        if len(self.line_features) > 10:
            raise ValueError("MultiLineDatasetVisualizer supports up to 10 line features")
        color_values = sorted(df[self.line_features[0]].unique())
        colors = dict(zip(color_values, sns.color_palette(self.palette, len(color_values))))
        style_values = sorted(df[self.line_features[1]].unique()) if len(self.line_features) > 1 else []
        if len(style_values) > len(line_styles):
            raise ValueError("Too many values for the second line feature")
        styles = dict(zip(style_values, line_styles))
        marker_values = sorted(df[self.line_features[2:]].drop_duplicates().itertuples(index=False, name=None)) if len(self.line_features) > 2 else []
        if len(marker_values) > len(markers):
            raise ValueError("Too many value combinations for marker line features")
        marker_by_value = dict(zip(marker_values, markers))
        horizontal_colors = sns.color_palette(self.horizontal_lines_palette, len(self.horizontal_lines))
        legend_handles = None
        legend_labels = None
        for row_idx, y_grid_value in enumerate(y_grid_values):
            for col_idx, x_grid_value in enumerate(x_grid_values):
                ax = axes[row_idx][col_idx]
                cell_df = df[
                    (df[self.x_grid_feature] == x_grid_value)
                    & (df[self.y_grid_feature] == y_grid_value)
                ]
                for line_values, line_df in cell_df.groupby(self.line_features):
                    line_values = line_values if isinstance(line_values, tuple) else (line_values,)
                    sns.lineplot(
                        data=line_df,
                        x=self.x_feature,
                        y=self.y_feature,
                        estimator="mean",
                        errorbar=("ci", 95) if self.show_confidence_interval else None,
                        color=colors[line_values[0]],
                        linestyle=styles[line_values[1]] if len(line_values) > 1 else "-",
                        marker=marker_by_value[line_values[2:]] if len(line_values) > 2 else "o",
                        linewidth=1.0,
                        markersize=4,
                        label=", ".join([
                            str(self.legend_renaming.get(feature, {}).get(value, value))
                            for feature, value in zip(self.line_features, line_values)
                        ]),
                        ax=ax,
                    )
                for (line_name, line_filters), color in zip(self.horizontal_lines.items(), horizontal_colors):
                    line_df = base_df
                    for feature, value in line_filters.items():
                        line_df = line_df[line_df[feature] == value]
                    line_y = line_df[self.y_feature].mean()
                    ax.axhline(line_y, linestyle="--", color=color)
                    ax.text(ax.get_xlim()[0], line_y, line_name, color=color, va="bottom", fontsize=7)
                ax.tick_params(axis="both", labelsize=8)
                ax.set_xlabel(self.feature_renaming.get(self.x_feature, self.x_feature), fontsize=9)
                ax.set_ylabel(self.feature_renaming.get(self.y_feature, self.y_feature), fontsize=9)
                if row_idx == 0:
                    ax.set_title(
                        f"{self.feature_renaming.get(self.x_grid_feature, self.x_grid_feature)}="
                        f"{self.legend_renaming.get(self.x_grid_feature, {}).get(x_grid_value, x_grid_value)}",
                        fontsize=12,
                        pad=10,
                    )
                if col_idx == 0:
                    ax.text(
                        -0.18,
                        0.5,
                        f"{self.feature_renaming.get(self.y_grid_feature, self.y_grid_feature)}="
                        f"{self.legend_renaming.get(self.y_grid_feature, {}).get(y_grid_value, y_grid_value)}",
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=12,
                    )
                if legend_handles is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
        if self.title is not None:
            fig.suptitle(self.title, fontsize=16)
        if legend_handles is not None:
            fig.legend(
                legend_handles,
                legend_labels,
                title=", ".join([self.feature_renaming.get(feature, feature) for feature in self.line_features]),
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=len(legend_labels),
            )
        fig.subplots_adjust(
            top=0.9 if self.title is not None else 0.96,
            bottom=0.28 if len(y_grid_values) == 1 else 0.18,
            right=0.96,
            wspace=0.25,
            hspace=0.35,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
        buf.seek(0)

        image = Image.open(buf)
        return image