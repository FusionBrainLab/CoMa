from typing import Any, Dict
import io

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from .image_data_visualizer import ImageDataVisualizer
from ..dataset_processor import DatasetProcessor

class FeatureRangeDatasetMultiSamplesVisualizer(ImageDataVisualizer):
    def __init__(self, *, feature: str,
                        dataset_key: str,
                        ranges_count: int,
                        samples_count: int,
                        id_col: str,
                        dataset_preprocessor: DatasetProcessor,
                        sample_visualizer: ImageDataVisualizer,
                        random_seed: int,
                        cell_size: int,
                        dpi: int) -> None:
        self.feature = feature
        self.dataset_key = dataset_key
        self.ranges_count = ranges_count
        self.samples_count = samples_count
        self.id_col = id_col
        self.dataset_preprocessor = dataset_preprocessor
        self.sample_visualizer = sample_visualizer
        self.random_seed = random_seed
        self.cell_size = cell_size
        self.dpi = dpi

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        dataset = data[self.dataset_key]
        pd_dataset = pd.DataFrame(dataset)
        pd_dataset = pd_dataset.sort_values(self.feature).reset_index(drop=True)
        feature_min = pd_dataset[self.feature].min()
        feature_max = pd_dataset[self.feature].max()
        feature_ranges = pd.interval_range(
            start=feature_min,
            end=feature_max,
            periods=self.ranges_count,
            closed="both"
        )
        cell_data = []

        for range_idx, feature_range in enumerate(feature_ranges):
            range_cell_data = []
            range_sample_ids = set()
            for expand_idx in range(self.ranges_count):
                min_idx = max(0, range_idx - expand_idx)
                max_idx = min(self.ranges_count - 1, range_idx + expand_idx)
                filtered_pd_dataset = pd_dataset[
                    (pd_dataset[self.feature] >= feature_ranges[min_idx].left) &
                    (pd_dataset[self.feature] <= feature_ranges[max_idx].right)
                ]
                filtered_pd_dataset = filtered_pd_dataset.sample(
                    frac=1,
                    random_state=self.random_seed + range_idx * self.ranges_count + expand_idx
                )

                for _, row in filtered_pd_dataset.iterrows():
                    sample_id = row[self.id_col]
                    if sample_id in range_sample_ids:
                        continue

                    try:
                        sampled_dataset = self.dataset_preprocessor(dataset=pd.DataFrame([row]).to_dict("list"))
                        sampled_dataset = pd.DataFrame(sampled_dataset)

                        sample = sampled_dataset.iloc[0].to_dict()
                        image = self.sample_visualizer(data=sample)
                        image = image.convert("RGB")
                        image.thumbnail((self.cell_size, self.cell_size))
                        range_cell_data.append({
                            "image":image,
                            "sample_id":sample[self.id_col],
                            "feature_value":row[self.feature]
                        })
                        range_sample_ids.add(sample_id)
                    except Exception:
                        continue

                    if len(range_cell_data) == self.samples_count:
                        break

                if len(range_cell_data) == self.samples_count:
                    break

            if len(range_cell_data) < self.samples_count:
                raise ValueError(f"Only {len(range_cell_data)} valid samples found for {feature_range}, expected {self.samples_count}")
            cell_data.append(range_cell_data)

        fig, axes = plt.subplots(
            self.samples_count,
            self.ranges_count,
            figsize=(self.ranges_count * 3, self.samples_count * 3),
            squeeze=False
        )

        for sample_idx in range(self.samples_count):
            for range_idx, feature_range in enumerate(feature_ranges):
                ax = axes[sample_idx][range_idx]
                ax.set_xticks([])
                ax.set_yticks([])

                if sample_idx == 0:
                    ax.set_title(f"[{feature_range.left:.2f}, {feature_range.right:.2f}]")
                if range_idx == 0:
                    ax.set_ylabel(f"sample {sample_idx + 1}")

                range_cell_data = cell_data[range_idx][sample_idx]
                ax.imshow(range_cell_data["image"])
                ax.set_xlabel(f"id: {range_cell_data['sample_id']}\n{self.feature}: {range_cell_data['feature_value']:.2f}")

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi)
        buf.seek(0)
        plt.close(fig)

        image = Image.open(buf)
        return image