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

class MultiBarplotDatasetVisualizer(ImageDataVisualizer):
    def __init__(self, *, dataset_key: str,
                        y_feature: str,
                        group_features: List[str],
                        title: Optional[str],
                        feature_renaming: Dict[str, str],
                        legend_renaming: Dict[Any, str],
                        fixed_features: Dict[str, Any],
                        horizontal_lines: Dict[str, Dict[str, Any]],
                        include_horizontal_line_samples: bool,
                        show_confidence_interval: bool,
                        palette: str,
                        horizontal_lines_palette: str,
                        bar_width: float) -> None:
        self.dataset_key = dataset_key
        self.y_feature = y_feature
        self.group_features = group_features
        self.title = title
        self.feature_renaming = feature_renaming
        self.legend_renaming = legend_renaming
        self.fixed_features = fixed_features
        self.horizontal_lines = horizontal_lines
        self.include_horizontal_line_samples = include_horizontal_line_samples
        self.show_confidence_interval = show_confidence_interval
        self.palette = palette
        self.horizontal_lines_palette = horizontal_lines_palette
        self.bar_width = bar_width

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

        sns.set_style("whitegrid")
        if len(self.group_features) == 0:
            raise ValueError("MultiBarplotDatasetVisualizer requires at least one group feature")
        if len(self.group_features) > 3:
            raise ValueError("MultiBarplotDatasetVisualizer supports up to 3 group features")
        x_feature = self.group_features[1] if len(self.group_features) > 1 else "_bar_group"
        hue_feature = "_legend_group"
        df[hue_feature] = df[self.group_features[0]].map(lambda value: self.legend_renaming.get(value, value))
        if len(self.group_features) == 1:
            df[x_feature] = ""
        y_values = df[self.y_feature].tolist()
        graph = sns.catplot(
            data=df,
            kind="bar",
            x=x_feature,
            y=self.y_feature,
            hue=hue_feature,
            col=self.group_features[2] if len(self.group_features) > 2 else None,
            estimator="mean",
            errorbar=("ci", 95) if self.show_confidence_interval else None,
            palette=self.palette,
            width=self.bar_width,
            height=6,
            aspect=1.5,
        )
        horizontal_colors = sns.color_palette(self.horizontal_lines_palette, len(self.horizontal_lines))
        for ax in graph.axes.flat:
            for (line_name, line_filters), color in zip(self.horizontal_lines.items(), horizontal_colors):
                line_df = base_df
                for feature, value in line_filters.items():
                    line_df = line_df[line_df[feature] == value]
                line_y = line_df[self.y_feature].mean()
                y_values.append(line_y)
                ax.axhline(line_y, linestyle="--", color=color)
                ax.text(ax.get_xlim()[0], line_y, line_name, color=color, va="bottom")
        y_min = min(y_values)
        y_max = max(y_values)
        y_padding = (y_max - y_min) * 0.1 if y_max > y_min else abs(y_min) * 0.01
        for ax in graph.axes.flat:
            ax.set_facecolor("#f2f2f2")
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

        graph.set_axis_labels(
            self.feature_renaming.get(x_feature, x_feature) if x_feature != "_bar_group" else "",
            self.feature_renaming.get(self.y_feature, self.y_feature),
        )
        if graph.legend is not None:
            graph.legend.set_title(self.feature_renaming.get(self.group_features[0], self.group_features[0]))
            sns.move_legend(
                graph,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=len(df[hue_feature].unique()),
            )
        if len(self.group_features) > 2:
            for ax, value in zip(graph.axes.flat, sorted(df[self.group_features[2]].unique())):
                ax.set_title(f"{self.feature_renaming.get(self.group_features[2], self.group_features[2])}={value}")
        if self.title is not None:
            graph.fig.suptitle(self.title, fontsize=16)
        graph.fig.subplots_adjust(top=0.88 if self.title is not None else 0.96, bottom=0.2)

        buf = io.BytesIO()
        graph.fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
        buf.seek(0)

        image = Image.open(buf)
        return image