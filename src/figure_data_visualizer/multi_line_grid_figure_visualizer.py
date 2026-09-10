from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from .figure_data_visualizer import FigureDataVisualizer
from .figure_layout_utils import (
    MATCHED_REFERENCE_DASH_PATTERN,
    annotate_reference_lines,
    apply_axes_style,
    apply_grid_labels,
    compute_reference_line_values,
    draw_figure_legend,
    draw_reference_lines,
    format_panel_title,
    limit_axis_ticks,
    matched_reference_line_colors,
    reference_line_colors,
)

LINE_STYLES = ["-", (0, (4.0, 1.6)), (0, (1.0, 1.4)), (0, (6.0, 1.6, 1.0, 1.6))]
MARKERS = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

class MultiLineGridFigureVisualizer(FigureDataVisualizer):
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
                        horizontal_lines_palette: str,
                        horizontal_lines_color_feature: Optional[str],
                        figure_width_inches: float,
                        figure_height_inches: float,
                        panel_face_color: str,
                        share_y_axis: bool,
                        line_width: float,
                        marker_size: float,
                        tick_font_size: float,
                        label_font_size: float,
                        panel_title_font_size: float,
                        legend_font_size: float,
                        legend_title: Optional[str],
                        legend_max_columns: int,
                        column_title_template: str,
                        row_title_template: str,
                        reference_font_size: float,
                        reference_label_mode: str,
                        reference_line_width: float,
                        max_x_ticks: int,
                        max_y_ticks: int,
                        integer_x_ticks: bool) -> None:
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
        self.horizontal_lines_color_feature = horizontal_lines_color_feature
        self.figure_width_inches = figure_width_inches
        self.figure_height_inches = figure_height_inches
        self.panel_face_color = panel_face_color
        self.share_y_axis = share_y_axis
        self.line_width = line_width
        self.marker_size = marker_size
        self.tick_font_size = tick_font_size
        self.label_font_size = label_font_size
        self.panel_title_font_size = panel_title_font_size
        self.legend_font_size = legend_font_size
        self.legend_title = legend_title
        self.legend_max_columns = legend_max_columns
        self.column_title_template = column_title_template
        self.row_title_template = row_title_template
        self.reference_font_size = reference_font_size
        self.reference_label_mode = reference_label_mode
        self.reference_line_width = reference_line_width
        self.max_x_ticks = max_x_ticks
        self.max_y_ticks = max_y_ticks
        self.integer_x_ticks = integer_x_ticks

    def __call__(self, *, data: Dict[str, Any]) -> Figure:
        dataset = data[self.dataset_key]
        base_dataframe = pd.DataFrame(dataset)
        dataframe = base_dataframe
        for feature, value in self.fixed_features.items():
            dataframe = dataframe[dataframe[feature] == value]
        if not self.include_horizontal_line_samples:
            for line_filters in self.horizontal_lines.values():
                mask = pd.Series(True, index=dataframe.index)
                for feature, value in line_filters.items():
                    mask = mask & (dataframe[feature] == value)
                dataframe = dataframe[~mask]
        if len(self.line_features) == 0:
            raise ValueError("MultiLineGridFigureVisualizer requires at least one line feature")
        if len(self.line_features) > 10:
            raise ValueError("MultiLineGridFigureVisualizer supports up to 10 line features")
        x_grid_values = sorted(dataframe[self.x_grid_feature].unique())
        y_grid_values = sorted(dataframe[self.y_grid_feature].unique())

        color_values = sorted(dataframe[self.line_features[0]].unique())
        colors = dict(zip(color_values, sns.color_palette(self.palette, len(color_values))))
        style_values = sorted(dataframe[self.line_features[1]].unique()) if len(self.line_features) > 1 else []
        if len(style_values) > len(LINE_STYLES):
            raise ValueError("Too many values for the second line feature")
        styles = dict(zip(style_values, LINE_STYLES))
        marker_values = (
            sorted(dataframe[self.line_features[2:]].drop_duplicates().itertuples(index=False, name=None))
            if len(self.line_features) > 2
            else []
        )
        if len(marker_values) > len(MARKERS):
            raise ValueError("Too many value combinations for marker line features")
        marker_by_value = dict(zip(marker_values, MARKERS))

        figure, axes = plt.subplots(
            len(y_grid_values),
            len(x_grid_values),
            figsize=(self.figure_width_inches, self.figure_height_inches),
            sharex="col",
            sharey=self.share_y_axis,
            squeeze=False,
            layout="constrained",
        )
        reference_values = compute_reference_line_values(
            dataframe=base_dataframe,
            horizontal_lines=self.horizontal_lines,
            value_feature=self.y_feature,
        )
        if self.horizontal_lines_color_feature is None:
            reference_colors = reference_line_colors(
                palette=self.horizontal_lines_palette,
                count=len(reference_values),
            )
            reference_dash_pattern = None
        else:
            reference_colors = matched_reference_line_colors(
                horizontal_lines=self.horizontal_lines,
                color_feature=self.horizontal_lines_color_feature,
                color_by_value=colors,
            )
            reference_dash_pattern = MATCHED_REFERENCE_DASH_PATTERN
        line_handles = None
        line_labels = None
        reference_handles = []
        for row_index, y_grid_value in enumerate(y_grid_values):
            for column_index, x_grid_value in enumerate(x_grid_values):
                axes_object = axes[row_index][column_index]
                cell_dataframe = dataframe[
                    (dataframe[self.x_grid_feature] == x_grid_value)
                    & (dataframe[self.y_grid_feature] == y_grid_value)
                ]
                for line_values, line_dataframe in cell_dataframe.groupby(self.line_features):
                    line_values = line_values if isinstance(line_values, tuple) else (line_values,)
                    sns.lineplot(
                        data=line_dataframe,
                        x=self.x_feature,
                        y=self.y_feature,
                        estimator="mean",
                        errorbar=("ci", 95) if self.show_confidence_interval else None,
                        color=colors[line_values[0]],
                        linestyle=styles[line_values[1]] if len(line_values) > 1 else "-",
                        marker=marker_by_value[line_values[2:]] if len(line_values) > 2 else "o",
                        linewidth=self.line_width,
                        markersize=self.marker_size,
                        markeredgecolor="white",
                        markeredgewidth=0.4,
                        label=", ".join([
                            str(self.legend_renaming.get(feature, {}).get(value, value))
                            for feature, value in zip(self.line_features, line_values)
                        ]),
                        ax=axes_object,
                    )
                if line_handles is None:
                    line_handles, line_labels = axes_object.get_legend_handles_labels()
                handles = draw_reference_lines(
                    axes_object=axes_object,
                    values=reference_values,
                    colors=reference_colors,
                    line_width=self.reference_line_width,
                    dash_pattern=reference_dash_pattern,
                )
                if len(reference_handles) == 0:
                    reference_handles = handles
                legend = axes_object.get_legend()
                if legend is not None:
                    legend.remove()
                apply_axes_style(
                    axes_object=axes_object,
                    tick_font_size=self.tick_font_size,
                    face_color=self.panel_face_color,
                    show_grid=True,
                )
                limit_axis_ticks(
                    axes_object=axes_object,
                    max_x_ticks=self.max_x_ticks,
                    max_y_ticks=self.max_y_ticks,
                    integer_x_ticks=self.integer_x_ticks,
                )

        if self.reference_label_mode == "inline":
            for axes_object in axes.flat:
                annotate_reference_lines(
                    axes_object=axes_object,
                    values=reference_values,
                    colors=reference_colors,
                    font_size=self.reference_font_size,
                )
        apply_grid_labels(
            figure=figure,
            axes=axes,
            x_label=self.feature_renaming.get(self.x_feature, self.x_feature),
            y_label=self.feature_renaming.get(self.y_feature, self.y_feature),
            column_titles=[
                format_panel_title(
                    template=self.column_title_template,
                    feature_label=self.feature_renaming.get(self.x_grid_feature, self.x_grid_feature),
                    value_label=str(self.legend_renaming.get(self.x_grid_feature, {}).get(value, value)),
                )
                for value in x_grid_values
            ],
            row_titles=[
                format_panel_title(
                    template=self.row_title_template,
                    feature_label=self.feature_renaming.get(self.y_grid_feature, self.y_grid_feature),
                    value_label=str(self.legend_renaming.get(self.y_grid_feature, {}).get(value, value)),
                )
                for value in y_grid_values
            ],
            label_font_size=self.label_font_size,
            panel_title_font_size=self.panel_title_font_size,
        )

        legend_handles = list(line_handles) if line_handles is not None else []
        legend_labels = list(line_labels) if line_labels is not None else []
        if self.reference_label_mode == "legend":
            legend_handles = legend_handles + reference_handles
            legend_labels = legend_labels + list(reference_values.keys())
        draw_figure_legend(
            figure=figure,
            handles=legend_handles,
            labels=legend_labels,
            title=self.legend_title,
            font_size=self.legend_font_size,
            max_columns=self.legend_max_columns,
        )
        if self.title is not None:
            figure.suptitle(self.title, fontsize=self.panel_title_font_size + 1.0)
        return figure
