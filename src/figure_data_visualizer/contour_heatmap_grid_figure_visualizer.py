from typing import Any, Dict, Optional

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .figure_data_visualizer import FigureDataVisualizer
from .figure_layout_utils import (
    annotation_text_color,
    apply_axes_style,
    apply_grid_labels,
    format_annotation_value,
    format_panel_title,
)

class ContourHeatmapGridFigureVisualizer(FigureDataVisualizer):
    def __init__(self, *, dataset_key: str,
                        x_feature: str,
                        y_feature: str,
                        value_feature: str,
                        x_grid_feature: str,
                        y_grid_feature: str,
                        title: Optional[str],
                        feature_renaming: Dict[str, str],
                        palette: str,
                        fixed_features: Dict[str, Any],
                        figure_width_inches: float,
                        figure_height_inches: float,
                        contour_levels: int,
                        smoothing_resolution: int,
                        value_format: str,
                        strip_annotation_leading_zero: bool,
                        show_value_annotations: bool,
                        tick_font_size: float,
                        label_font_size: float,
                        panel_title_font_size: float,
                        annotation_font_size: float,
                        column_title_template: str,
                        row_title_template: str,
                        colorbar_width_fraction: float) -> None:
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
        self.figure_width_inches = figure_width_inches
        self.figure_height_inches = figure_height_inches
        self.contour_levels = contour_levels
        self.smoothing_resolution = smoothing_resolution
        self.value_format = value_format
        self.strip_annotation_leading_zero = strip_annotation_leading_zero
        self.show_value_annotations = show_value_annotations
        self.tick_font_size = tick_font_size
        self.label_font_size = label_font_size
        self.panel_title_font_size = panel_title_font_size
        self.annotation_font_size = annotation_font_size
        self.column_title_template = column_title_template
        self.row_title_template = row_title_template
        self.colorbar_width_fraction = colorbar_width_fraction

    def __call__(self, *, data: Dict[str, Any]) -> Figure:
        dataset = data[self.dataset_key]
        dataframe = pd.DataFrame(dataset)
        for feature, value in self.fixed_features.items():
            dataframe = dataframe[dataframe[feature] == value]
        x_grid_values = sorted(dataframe[self.x_grid_feature].unique())
        y_grid_values = sorted(dataframe[self.y_grid_feature].unique())
        value_minimum = dataframe[self.value_feature].min()
        value_maximum = dataframe[self.value_feature].max()
        heatmap_dataframes = {}
        for y_grid_value in y_grid_values:
            for x_grid_value in x_grid_values:
                cell_dataframe = dataframe[
                    (dataframe[self.x_grid_feature] == x_grid_value)
                    & (dataframe[self.y_grid_feature] == y_grid_value)
                ]
                heatmap_dataframes[(y_grid_value, x_grid_value)] = cell_dataframe.groupby(
                    [self.y_feature, self.x_feature],
                    as_index=False,
                )[self.value_feature].mean().pivot(
                    index=self.y_feature,
                    columns=self.x_feature,
                    values=self.value_feature,
                ).sort_index().sort_index(axis=1)

        figure, axes = plt.subplots(
            len(y_grid_values),
            len(x_grid_values),
            figsize=(self.figure_width_inches, self.figure_height_inches),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        normalization = colors.Normalize(vmin=value_minimum, vmax=value_maximum)
        color_map = plt.get_cmap(self.palette)
        for row_index, y_grid_value in enumerate(y_grid_values):
            for column_index, x_grid_value in enumerate(x_grid_values):
                axes_object = axes[row_index][column_index]
                heatmap_dataframe = heatmap_dataframes[(y_grid_value, x_grid_value)]
                column_count = len(heatmap_dataframe.columns)
                row_count = len(heatmap_dataframe.index)
                x_centers = np.arange(column_count) + 0.5
                y_centers = np.arange(row_count) + 0.5
                x_smooth = np.linspace(0, column_count, self.smoothing_resolution)
                y_smooth = np.linspace(0, row_count, self.smoothing_resolution)
                value_by_x = np.array([np.interp(x_smooth, x_centers, row) for row in heatmap_dataframe.values])
                value_smooth = np.array([
                    np.interp(y_smooth, y_centers, value_by_x[:, index]) for index in range(len(x_smooth))
                ]).T
                levels = np.linspace(np.nanmin(value_smooth), np.nanmax(value_smooth), self.contour_levels)
                contour_set = axes_object.contourf(
                    x_smooth,
                    y_smooth,
                    value_smooth,
                    levels=levels,
                    cmap=self.palette,
                    norm=normalization,
                )
                # Vector backends leave hairline seams between abutting filled bands.
                contour_set.set_edgecolor("face")
                if self.show_value_annotations:
                    for y_index, y_value in enumerate(heatmap_dataframe.index):
                        for x_index, x_value in enumerate(heatmap_dataframe.columns):
                            cell_value = heatmap_dataframe.loc[y_value, x_value]
                            axes_object.text(
                                x_index + 0.5,
                                y_index + 0.5,
                                format_annotation_value(
                                    value=cell_value,
                                    value_format=self.value_format,
                                    strip_leading_zero=self.strip_annotation_leading_zero,
                                ),
                                horizontalalignment="center",
                                verticalalignment="center",
                                fontsize=self.annotation_font_size,
                                color=annotation_text_color(background=color_map(normalization(cell_value))),
                            )
                axes_object.set_xlim(0, column_count)
                axes_object.set_ylim(row_count, 0)
                axes_object.set_xticks(x_centers)
                axes_object.set_yticks(y_centers)
                axes_object.set_xticklabels(heatmap_dataframe.columns)
                axes_object.set_yticklabels(heatmap_dataframe.index)
                axes_object.set_xticks(np.arange(column_count + 1), minor=True)
                axes_object.set_yticks(np.arange(row_count + 1), minor=True)
                apply_axes_style(
                    axes_object=axes_object,
                    tick_font_size=self.tick_font_size,
                    face_color="white",
                    show_grid=False,
                )
                axes_object.grid(which="minor", color="white", linewidth=0.3, alpha=0.35)
                axes_object.set_axisbelow(False)

        apply_grid_labels(
            figure=figure,
            axes=axes,
            x_label=self.feature_renaming.get(self.x_feature, self.x_feature),
            y_label=self.feature_renaming.get(self.y_feature, self.y_feature),
            column_titles=[
                format_panel_title(
                    template=self.column_title_template,
                    feature_label=self.feature_renaming.get(self.x_grid_feature, self.x_grid_feature),
                    value_label=str(value),
                )
                for value in x_grid_values
            ],
            row_titles=[
                format_panel_title(
                    template=self.row_title_template,
                    feature_label=self.feature_renaming.get(self.y_grid_feature, self.y_grid_feature),
                    value_label=str(value),
                )
                for value in y_grid_values
            ],
            label_font_size=self.label_font_size,
            panel_title_font_size=self.panel_title_font_size,
        )
        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(norm=normalization, cmap=self.palette),
            ax=axes.ravel().tolist(),
            fraction=self.colorbar_width_fraction,
            pad=0.015,
        )
        if colorbar.solids is not None:
            # Keeps the pgf self-contained instead of referencing a sidecar bitmap.
            colorbar.solids.set_rasterized(False)
            colorbar.solids.set_edgecolor("face")
        colorbar.set_label(
            self.feature_renaming.get(self.value_feature, self.value_feature),
            fontsize=self.label_font_size,
        )
        colorbar.ax.tick_params(labelsize=self.tick_font_size, length=2.0, width=0.5, pad=1.5)
        colorbar.outline.set_linewidth(0.5)
        if self.title is not None:
            figure.suptitle(self.title, fontsize=self.panel_title_font_size + 1.0)
        return figure
