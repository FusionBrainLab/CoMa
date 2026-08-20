from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import seaborn as sns
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MaxNLocator

REFERENCE_DASH_PATTERNS = [(6.0, 2.0), (2.0, 1.6), (7.0, 1.6, 1.0, 1.6), (1.0, 1.4)]
MATCHED_REFERENCE_DASH_PATTERN = (5.0, 1.4, 1.0, 1.4)
GRID_COLOR = "#cfcfcf"
SPINE_COLOR = "#b8b8b8"
TICK_STEPS = [1, 2, 2.5, 5, 10]

def compute_reference_line_values(*, dataframe: Any,
                                    horizontal_lines: Dict[str, Dict[str, Any]],
                                    value_feature: str) -> Dict[str, float]:
    values = {}
    for line_name, line_filters in horizontal_lines.items():
        line_dataframe = dataframe
        for feature, value in line_filters.items():
            line_dataframe = line_dataframe[line_dataframe[feature] == value]
        values[line_name] = float(line_dataframe[value_feature].mean())
    return values

def reference_line_colors(*, palette: str, count: int) -> List[Tuple[float, float, float]]:
    return [tuple(color) for color in sns.color_palette(palette, count)]

def feature_value_palette(*, horizontal_lines: Dict[str, Dict[str, Any]],
                            color_feature: str,
                            palette: str) -> Dict[Any, Tuple[float, float, float]]:
    values = sorted({
        line_filters[color_feature]
        for line_filters in horizontal_lines.values()
        if color_feature in line_filters
    })
    return {value: tuple(color) for value, color in zip(values, sns.color_palette(palette, len(values)))}

def matched_reference_line_colors(*, horizontal_lines: Dict[str, Dict[str, Any]],
                                    color_feature: str,
                                    color_by_value: Dict[Any, Any]) -> List[Tuple[float, float, float]]:
    matched_colors = []
    for line_name, line_filters in horizontal_lines.items():
        if color_feature not in line_filters:
            raise ValueError(
                f"Horizontal line '{line_name}' has no '{color_feature}' filter to match a series color"
            )
        value = line_filters[color_feature]
        if value not in color_by_value:
            raise ValueError(
                f"Horizontal line '{line_name}' refers to {color_feature}={value}, which has no plotted series"
            )
        matched_colors.append(tuple(color_by_value[value]))
    return matched_colors

def uniform_reference_line_colors(*, color: str, count: int) -> List[str]:
    return [color] * count

def draw_reference_lines(*, axes_object: Any,
                           values: Dict[str, float],
                           colors: List[Any],
                           line_width: float,
                           dash_pattern: Optional[Tuple[float, ...]]) -> List[Any]:
    handles = []
    for index, ((_, value), color) in enumerate(zip(values.items(), colors)):
        handles.append(axes_object.axhline(
            value,
            color=color,
            linewidth=line_width,
            dashes=dash_pattern or REFERENCE_DASH_PATTERNS[index % len(REFERENCE_DASH_PATTERNS)],
            zorder=1.6,
        ))
    return handles

def reference_span_positions(*, axes_object: Any,
                               horizontal_lines: Dict[str, Dict[str, Any]],
                               span_feature: str,
                               value_renaming: Dict[Any, str]) -> Dict[str, float]:
    position_by_label = {
        text.get_text(): position
        for position, text in zip(axes_object.get_xticks(), axes_object.get_xticklabels())
    }
    positions = {}
    for line_name, line_filters in horizontal_lines.items():
        value = line_filters.get(span_feature)
        label = str(value_renaming.get(value, value))
        if label in position_by_label:
            positions[line_name] = float(position_by_label[label])
    return positions

def draw_reference_spans(*, axes_object: Any,
                           values: Dict[str, float],
                           positions: Dict[str, float],
                           colors: List[Any],
                           line_width: float,
                           span_width: float,
                           dash_pattern: Optional[Tuple[float, ...]]) -> List[Any]:
    handles = []
    for index, ((line_name, value), color) in enumerate(zip(values.items(), colors)):
        if line_name not in positions:
            continue
        center = positions[line_name]
        handles.append(axes_object.plot(
            [center - span_width / 2.0, center + span_width / 2.0],
            [value, value],
            color=color,
            linewidth=line_width,
            dashes=dash_pattern or REFERENCE_DASH_PATTERNS[index % len(REFERENCE_DASH_PATTERNS)],
            solid_capstyle="butt",
            # Spans cross the bars they annotate, so each dash carries a halo to stay legible
            # against both the panel background and a saturated bar fill.
            path_effects=[withStroke(linewidth=line_width + 1.1, foreground="white")],
            zorder=2.6,
        )[0])
    return handles

def annotate_reference_lines(*, axes_object: Any,
                               values: Dict[str, float],
                               colors: List[Tuple[float, float, float]],
                               font_size: float) -> None:
    y_min, y_max = axes_object.get_ylim()
    if y_max <= y_min:
        return
    axes_height_points = axes_object.get_position().height * axes_object.get_figure().get_figheight() * 72.0
    minimum_gap = font_size * 1.6 / max(axes_height_points, 1.0)
    fractions = [(value - y_min) / (y_max - y_min) for value in values.values()]
    placed = list(fractions)
    previous = None
    for index in np.argsort(fractions):
        current = placed[index] if previous is None else max(placed[index], previous + minimum_gap)
        placed[index] = current
        previous = current
    for (line_name, _), color, fraction in zip(values.items(), colors, placed):
        axes_object.text(
            0.99,
            min(max(fraction, 0.0), 1.0),
            line_name,
            transform=axes_object.transAxes,
            color=color,
            fontsize=font_size,
            horizontalalignment="right",
            verticalalignment="center",
            zorder=3.0,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.9},
        )

def apply_axes_style(*, axes_object: Any,
                       tick_font_size: float,
                       face_color: str,
                       show_grid: bool) -> None:
    axes_object.set_facecolor(face_color)
    if show_grid:
        axes_object.grid(True, color=GRID_COLOR, linewidth=0.4)
    else:
        axes_object.grid(False)
    axes_object.set_axisbelow(True)
    axes_object.tick_params(axis="both", labelsize=tick_font_size, length=2.0, width=0.5, pad=1.5)
    for spine in axes_object.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(SPINE_COLOR)

def limit_axis_ticks(*, axes_object: Any,
                       max_x_ticks: Optional[int],
                       max_y_ticks: Optional[int],
                       integer_x_ticks: bool) -> None:
    if max_x_ticks is not None:
        axes_object.xaxis.set_major_locator(
            MaxNLocator(nbins=max_x_ticks, integer=integer_x_ticks, steps=TICK_STEPS)
        )
    if max_y_ticks is not None:
        axes_object.yaxis.set_major_locator(MaxNLocator(nbins=max_y_ticks, steps=TICK_STEPS))

def apply_grid_labels(*, figure: Any,
                        axes: Any,
                        x_label: str,
                        y_label: str,
                        column_titles: List[str],
                        row_titles: List[str],
                        label_font_size: float,
                        panel_title_font_size: float,
                        row_title_offset: float = -0.62) -> None:
    row_count = len(axes)
    column_count = len(axes[0])
    use_row_titles = len(row_titles) == row_count
    for row_index in range(row_count):
        for column_index in range(column_count):
            axes_object = axes[row_index][column_index]
            axes_object.set_xlabel(
                x_label if row_index == row_count - 1 else "",
                fontsize=label_font_size,
                labelpad=2.0,
            )
            if column_index == 0:
                # Every row keeps its own copy of the axis quantity, matching the original
                # figures instead of merging it into one figure-wide label. A merged label
                # would sit at a single height and stop lining up once rows differ in size.
                axes_object.set_ylabel(y_label, fontsize=label_font_size, labelpad=5.0)
                if use_row_titles:
                    # The grouping label (e.g. "Train Type=Genuine") belongs outside the
                    # axis quantity, so it is placed further left in the same axes-fraction
                    # space rather than replacing the ylabel.
                    axes_object.text(
                        row_title_offset,
                        0.5,
                        row_titles[row_index],
                        transform=axes_object.transAxes,
                        rotation=90,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=panel_title_font_size,
                    )
            else:
                axes_object.set_ylabel("")
            if row_index == 0 and column_index < len(column_titles):
                axes_object.set_title(column_titles[column_index], fontsize=panel_title_font_size, pad=3.0)

def format_panel_title(*, template: str, feature_label: str, value_label: str) -> str:
    return template.format(feature=feature_label, value=value_label)

def format_annotation_value(*, value: float, value_format: str, strip_leading_zero: bool) -> str:
    label = format(value, value_format)
    if strip_leading_zero and label.startswith("0."):
        return label[1:]
    if strip_leading_zero and label.startswith("-0."):
        return "-" + label[2:]
    return label

def annotation_text_color(*, background: Tuple[float, float, float, float]) -> str:
    luminance = 0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
    return "white" if luminance < 0.55 else "#1a1a1a"

def legend_column_count(*, label_count: int, max_columns: int) -> int:
    return max(min(label_count, max_columns), 1)

def draw_figure_legend(*, figure: Any,
                         handles: List[Any],
                         labels: List[str],
                         title: Optional[str],
                         font_size: float,
                         max_columns: int) -> None:
    if len(handles) == 0:
        return
    figure.legend(
        handles,
        labels,
        title=title,
        loc="outside lower center",
        ncol=legend_column_count(label_count=len(labels), max_columns=max_columns),
        fontsize=font_size,
        title_fontsize=font_size,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
        handletextpad=0.5,
        borderaxespad=0.0,
    )
