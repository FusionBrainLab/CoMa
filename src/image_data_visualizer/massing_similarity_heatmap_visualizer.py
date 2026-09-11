from typing import Any, Dict, List, Optional
import io
import re

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns

from .image_data_visualizer import ImageDataVisualizer

class MassingSimilarityHeatmapVisualizer(ImageDataVisualizer):
    def __init__(self, *, dataset_key: str,
                        path_feature: Optional[str],
                        value_feature: str,
                        value_name: str,
                        model_agnostic: bool,
                        palette: str,
                        value_range: Optional[List[float]],
                        symmetric_value_range: bool,
                        show_positive_sign: bool,
                        figure_width: float,
                        row_height: float,
                        dpi: int,
                        font_size: int = 10) -> None:
        self.dataset_key = dataset_key
        self.path_feature = path_feature
        self.value_feature = value_feature
        self.value_name = value_name
        self.model_agnostic = model_agnostic
        self.palette = palette
        self.value_range = value_range
        self.symmetric_value_range = symmetric_value_range
        self.show_positive_sign = show_positive_sign
        self.figure_width = figure_width
        self.row_height = row_height
        self.dpi = dpi
        self.font_size = font_size

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        df = pd.DataFrame(data[self.dataset_key])
        if self.path_feature is not None:
            path_pattern = re.compile(
                r"([^/]+)/([0-9]+)B/1d([0-9]+)_2d([0-9]+)_3d([0-9]+)"
            )
            path_features = df[self.path_feature].str.extract(path_pattern)
            path_features.columns = [
                "train_type",
                "model_size",
                "context_1d_count",
                "context_2d_count",
                "context_3d_count",
            ]
            numeric_cols = [
                "model_size",
                "context_1d_count",
                "context_2d_count",
                "context_3d_count",
            ]
            path_features[numeric_cols] = path_features[numeric_cols].astype(int)
            df = pd.concat([df, path_features], axis=1)
        df = df[df["context_1d_count"] > 0]

        model_sizes = (
            ["All models"]
            if self.model_agnostic
            else sorted(df["model_size"].unique())
        )
        fig = plt.figure(
            figsize=(
                self.figure_width,
                self.row_height * len(model_sizes)
            )
        )
        grid = fig.add_gridspec(
            len(model_sizes),
            2,
            width_ratios=[10, 1.4],
            hspace=0.38 * self.font_size / 10,
            wspace=0.12,
            bottom=0.14,
        )
        if self.symmetric_value_range:
            max_value = max(
                abs(df[self.value_feature].min()),
                abs(df[self.value_feature].max()),
                1e-12,
            )
            norm = colors.Normalize(vmin=-max_value, vmax=max_value)
        else:
            assert self.value_range is not None
            norm = colors.Normalize(
                vmin=self.value_range[0],
                vmax=self.value_range[1]
            )
        sns.set_style("whitegrid")

        for row_idx, model_size in enumerate(model_sizes):
            model_df = (
                df
                if self.model_agnostic
                else df[df["model_size"] == model_size]
            )
            multimodal_df = model_df[
                model_df["train_type"] == "multimodal_context"
            ].copy()
            multimodal_df["other_context"] = multimodal_df.apply(
                lambda row: (
                    f"2D={row['context_2d_count']}, "
                    f"3D={row['context_3d_count']}"
                ),
                axis=1,
            )
            multimodal_heatmap = multimodal_df.pivot_table(
                index="context_1d_count",
                columns="other_context",
                values=self.value_feature,
                aggfunc="mean",
            ).sort_index()

            unimodal_df = model_df[
                (model_df["train_type"] == "unimodal_context")
                & (model_df["context_2d_count"] == 0)
                & (model_df["context_3d_count"] == 0)
            ]
            unimodal_heatmap = unimodal_df.pivot_table(
                index="context_1d_count",
                columns="train_type",
                values=self.value_feature,
                aggfunc="mean",
            ).sort_index()
            unimodal_heatmap.columns = ["1D only"]

            multimodal_ax = fig.add_subplot(grid[row_idx, 0])
            unimodal_ax = fig.add_subplot(grid[row_idx, 1])
            for ax, heatmap_df in [
                (multimodal_ax, multimodal_heatmap),
                (unimodal_ax, unimodal_heatmap),
            ]:
                annotations = (
                    heatmap_df.map(
                        lambda value: f"{value:+.3f}".replace(
                            "-", "\N{MINUS SIGN}"
                        )
                    )
                    if self.show_positive_sign
                    else True
                )
                sns.heatmap(
                    heatmap_df,
                    annot=annotations,
                    fmt="" if self.show_positive_sign else ".3f",
                    cmap=self.palette,
                    norm=norm,
                    cbar=False,
                    linewidths=0.5,
                    linecolor="white",
                    annot_kws={"fontsize": self.font_size},
                    ax=ax,
                )
                ax.tick_params(
                    axis="x",
                    top=True,
                    labeltop=True,
                    bottom=False,
                    labelbottom=False,
                    labelsize=self.font_size,
                )
                ax.tick_params(axis="y", labelsize=self.font_size)
                ax.set_xticklabels(
                    ax.get_xticklabels(),
                    rotation=45,
                    ha="left",
                    rotation_mode="anchor",
                )
                ax.set_yticklabels(
                    ax.get_yticklabels(),
                    rotation=0,
                )
                ax.set_xlabel("")
                ax.set_ylabel("")

            multimodal_ax.set_xlabel(
                "Multimodal train",
                labelpad=12,
                fontsize=self.font_size,
            )
            unimodal_ax.set_xlabel(
                "Unimodal train",
                labelpad=12,
                fontsize=self.font_size,
            )
            multimodal_ax.set_ylabel(
                (
                    "1D context count"
                    if self.model_agnostic
                    else f"Model {model_size}B\n1D context count"
                ),
                fontsize=self.font_size,
            )
            unimodal_ax.tick_params(axis="y", labelleft=False)

        colorbar_ax = fig.add_axes([0.125, 0.04, 0.775, 0.02])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=self.palette),
            cax=colorbar_ax,
            orientation="horizontal",
        )
        colorbar.set_label(self.value_name, fontsize=self.font_size)
        colorbar.ax.tick_params(labelsize=self.font_size)

        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=self.dpi,
            bbox_inches="tight"
        )
        plt.close(fig)
        buffer.seek(0)
        return Image.open(buffer)
