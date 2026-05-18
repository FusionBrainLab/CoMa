from typing import Any, Dict, List
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .core.base import Function

class DistributionsVisualizer(Function):
    def __init__(self, *, save_path: str,
                        col: str, 
                        fontsize: int,
                        palette: List[str],
                        log_scale: bool) -> None:
        self.save_path = save_path
        self.col = col
        self.fontsize = fontsize
        self.palette = palette
        self.log_scale = log_scale

    def __call__(self, *, datasets: Dict[str, Dict[str, List[Any]]]) -> Any:
        pd_datasets = {k: pd.DataFrame(v) for k, v in datasets.items()}
        plots_data = {k: {} for k in pd_datasets.keys()}
        for name, dataset in pd_datasets.items():
            plots_data[name][self.col] = dataset[self.col].values.tolist()

        plot_df_data = {
            self.col: [],
            "name": []
        }
        for name in plots_data.keys():
            length = len(plots_data[name][self.col])
            data = plots_data[name][self.col]
            plot_df_data[self.col].extend(data)
            plot_df_data["name"].extend([name] * length)
        plot_df = pd.DataFrame(plot_df_data)

        fontsize = self.fontsize
        palette = self.palette
        sns.set_style("whitegrid") 
        for name in plots_data["train"].keys():
            plt.figure(figsize=(10, 6))

            ax = sns.displot(data=plot_df, x=name, hue='name', kind='kde', 
                fill=True, common_norm=False, palette=palette,
                alpha=0.2, height=6, aspect=1.5, log_scale=self.log_scale)

            plt.xlabel('Values', fontsize=fontsize)
            plt.ylabel('Density', fontsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
            plt.setp(ax.legend.get_title(), fontsize=fontsize)
            plt.setp(ax.legend.get_texts(), fontsize=fontsize)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(self.save_path, dpi=300, bbox_inches='tight')