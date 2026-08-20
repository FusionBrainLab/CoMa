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
    compute_reference_line_values,
    draw_figure_legend,
    draw_reference_lines,
    draw_reference_spans,
    feature_value_palette,
    format_panel_title,
    limit_axis_ticks,
    matched_reference_line_colors,
    reference_line_colors,
    reference_span_positions,
    uniform_reference_line_colors,
)

class MultiBarplotFigureVisualizer(FigureDataVisualizer):
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
                        horizontal_lines_color: Optional[str],
                        horizontal_lines_color_feature: Optional[str],
                        horizontal_lines_span_feature: Optional[str],
                        horizontal_lines_span_width: float,
                        horizontal_lines_legend_label: Optional[str],
                        bar_width: float,
                        figure_width_inches: float,
                        figure_height_inches: float,
                        panel_face_color: str,
                        tick_font_size: float,
                        label_font_size: float,
                        panel_title_font_size: float,
                        legend_font_size: float,
                        legend_title: Optional[str],
                        legend_max_columns: int,
                        panel_title_template: str,
                        reference_font_size: float,
                        reference_label_mode: str,
                        reference_line_width: float,
                        max_y_ticks: int) -> None:
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
        self.horizontal_lines_color = horizontal_lines_color
        self.horizontal_lines_color_feature = horizontal_lines_color_feature
        self.horizontal_lines_span_feature = horizontal_lines_span_feature
        self.horizontal_lines_span_width = horizontal_lines_span_width
        self.horizontal_lines_legend_label = horizontal_lines_legend_label
        self.bar_width = bar_width
        self.figure_width_inches = figure_width_inches
        self.figure_height_inches = figure_height_inches
        self.panel_face_color = panel_face_color
        self.tick_font_size = tick_font_size
        self.label_font_size = label_font_size
        self.panel_title_font_size = panel_title_font_size
        self.legend_font_size = legend_font_size
        self.legend_title = legend_title
        self.legend_max_columns = legend_max_columns
        self.panel_title_template = panel_title_template
        self.reference_font_size = reference_font_size
        self.reference_label_mode = reference_label_mode
        self.reference_line_width = reference_line_width
        self.max_y_ticks = max_y_ticks

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
        if len(self.group_features) == 0:
            raise ValueError("MultiBarplotFigureVisualizer requires at least one group feature")
        if len(self.group_features) > 3:
            raise ValueError("MultiBarplotFigureVisualizer supports up to 3 group features")

        hue_feature = "_legend_group"
        dataframe = dataframe.copy()
        dataframe[hue_feature] = dataframe[self.group_features[0]].map(
            lambda value: self.legend_renaming.get(value, value)
        )
        if len(self.group_features) > 1:
            x_feature = self.group_features[1]
            # Relabel after sorting the raw values so that numeric sizes keep their natural
            # order instead of being sorted as the renamed strings.
            x_order = [
                str(self.legend_renaming.get(value, value))
                for value in sorted(dataframe[x_feature].unique())
            ]
            dataframe[x_feature] = dataframe[x_feature].map(
                lambda value: str(self.legend_renaming.get(value, value))
            )
        else:
            x_feature = "_bar_group"
            dataframe[x_feature] = ""
            x_order = [""]
        panel_feature = self.group_features[2] if len(self.group_features) > 2 else None
        panel_values = sorted(dataframe[panel_feature].unique()) if panel_feature is not None else [None]

        figure, axes = plt.subplots(
            1,
            len(panel_values),
            figsize=(self.figure_width_inches, self.figure_height_inches),
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        reference_values = compute_reference_line_values(
            dataframe=base_dataframe,
            horizontal_lines=self.horizontal_lines,
            value_feature=self.y_feature,
        )
        if self.horizontal_lines_color is not None:
            reference_colors = uniform_reference_line_colors(
                color=self.horizontal_lines_color,
                count=len(reference_values),
            )
            reference_dash_pattern = MATCHED_REFERENCE_DASH_PATTERN
        elif self.horizontal_lines_color_feature is not None:
            reference_colors = matched_reference_line_colors(
                horizontal_lines=self.horizontal_lines,
                color_feature=self.horizontal_lines_color_feature,
                color_by_value=feature_value_palette(
                    horizontal_lines=self.horizontal_lines,
                    color_feature=self.horizontal_lines_color_feature,
                    palette=self.horizontal_lines_palette,
                ),
            )
            reference_dash_pattern = MATCHED_REFERENCE_DASH_PATTERN
        else:
            reference_colors = reference_line_colors(
                palette=self.horizontal_lines_palette,
                count=len(reference_values),
            )
            reference_dash_pattern = None
        bar_handles = None
        bar_labels = None
        reference_handles = []
        for panel_index, panel_value in enumerate(panel_values):
            axes_object = axes[0][panel_index]
            panel_dataframe = dataframe if panel_value is None else dataframe[dataframe[panel_feature] == panel_value]
            sns.barplot(
                data=panel_dataframe,
                x=x_feature,
                y=self.y_feature,
                hue=hue_feature,
                order=x_order,
                estimator="mean",
                errorbar=("ci", 95) if self.show_confidence_interval else None,
                palette=self.palette,
                width=self.bar_width,
                capsize=0.12,
                err_kws={"linewidth": 0.7, "color": "#5a5a5a"},
                linewidth=0.0,
                legend=panel_index == 0,
                ax=axes_object,
            )
            if self.horizontal_lines_span_feature is None:
                handles = draw_reference_lines(
                    axes_object=axes_object,
                    values=reference_values,
                    colors=reference_colors,
                    line_width=self.reference_line_width,
                    dash_pattern=reference_dash_pattern,
                )
            else:
                handles = draw_reference_spans(
                    axes_object=axes_object,
                    values=reference_values,
                    positions=reference_span_positions(
                        axes_object=axes_object,
                        horizontal_lines=self.horizontal_lines,
                        span_feature=self.horizontal_lines_span_feature,
                        value_renaming=self.legend_renaming,
                    ),
                    colors=reference_colors,
                    line_width=self.reference_line_width,
                    span_width=self.horizontal_lines_span_width,
                    dash_pattern=reference_dash_pattern,
                )
            if panel_index == 0:
                legend = axes_object.get_legend()
                bar_handles = list(legend.legend_handles)
                bar_labels = [text.get_text() for text in legend.get_texts()]
                legend.remove()
                reference_handles = handles
            apply_axes_style(
                axes_object=axes_object,
                tick_font_size=self.tick_font_size,
                face_color=self.panel_face_color,
                show_grid=True,
            )
            limit_axis_ticks(
                axes_object=axes_object,
                max_x_ticks=None,
                max_y_ticks=self.max_y_ticks,
                integer_x_ticks=False,
            )
            axes_object.set_xlabel(
                self.feature_renaming.get(x_feature, x_feature) if x_feature != "_bar_group" else "",
                fontsize=self.label_font_size,
                labelpad=2.0,
            )
            axes_object.set_ylabel(
                self.feature_renaming.get(self.y_feature, self.y_feature) if panel_index == 0 else "",
                fontsize=self.label_font_size,
                labelpad=2.0,
            )
            if panel_feature is not None:
                axes_object.set_title(
                    format_panel_title(
                        template=self.panel_title_template,
                        feature_label=self.feature_renaming.get(panel_feature, panel_feature),
                        value_label=str(self.legend_renaming.get(panel_value, panel_value)),
                    ),
                    fontsize=self.panel_title_font_size,
                    pad=3.0,
                )

        value_minimum = min([*dataframe[self.y_feature].tolist(), *reference_values.values()])
        value_maximum = max([*dataframe[self.y_feature].tolist(), *reference_values.values()])
        padding = (value_maximum - value_minimum) * 0.08 if value_maximum > value_minimum else abs(value_minimum) * 0.01
        for axes_object in axes.flat:
            axes_object.set_ylim(value_minimum - padding, value_maximum + padding)
        if self.reference_label_mode == "inline":
            for axes_object in axes.flat:
                annotate_reference_lines(
                    axes_object=axes_object,
                    values=reference_values,
                    colors=reference_colors,
                    font_size=self.reference_font_size,
                )

        legend_handles = list(bar_handles) if bar_handles is not None else []
        legend_labels = list(bar_labels) if bar_labels is not None else []
        if self.reference_label_mode == "legend" and self.horizontal_lines_legend_label is not None:
            legend_handles = legend_handles + reference_handles[:1]
            legend_labels = legend_labels + [self.horizontal_lines_legend_label]
        elif self.reference_label_mode == "legend":
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
