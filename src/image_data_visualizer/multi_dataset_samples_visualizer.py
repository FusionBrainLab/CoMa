from typing import Any, Dict, List
import io

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from .image_data_visualizer import ImageDataVisualizer
from ..dataset_processor import DatasetProcessor

class MultiDatasetSamplesVisualizer(ImageDataVisualizer):
    def __init__(self, *, dataset_keys: List[str],
                        samples_count: int,
                        id_col: str,
                        dataset_preprocessor: DatasetProcessor,
                        sample_visualizer: ImageDataVisualizer,
                        random_seed: int,
                        cell_size: int,
                        dpi: int) -> None:
        self.dataset_keys = dataset_keys
        self.samples_count = samples_count
        self.id_col = id_col
        self.dataset_preprocessor = dataset_preprocessor
        self.sample_visualizer = sample_visualizer
        self.random_seed = random_seed
        self.cell_size = cell_size
        self.dpi = dpi

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        pd_datasets = {
            dataset_key: pd.DataFrame(data[dataset_key])
            for dataset_key in self.dataset_keys
        }
        common_ids = set(pd_datasets[self.dataset_keys[0]][self.id_col])
        for dataset_key in self.dataset_keys[1:]:
            common_ids = common_ids & set(pd_datasets[dataset_key][self.id_col])
        common_ids = pd.Series(list(common_ids)).sample(frac=1, random_state=self.random_seed).tolist()
        cell_data = []

        for sample_id in common_ids:
            sample_cell_data = []
            try:
                for dataset_key in self.dataset_keys:
                    sampled_dataset = pd_datasets[dataset_key][pd_datasets[dataset_key][self.id_col] == sample_id].head(1)
                    sampled_dataset = self.dataset_preprocessor(dataset=sampled_dataset.to_dict("list"))
                    sampled_dataset = pd.DataFrame(sampled_dataset)

                    sample = sampled_dataset.iloc[0].to_dict()
                    image = self.sample_visualizer(data=sample)
                    image = image.convert("RGB")
                    image.thumbnail((self.cell_size, self.cell_size))
                    sample_cell_data.append({
                        "image":image,
                        "sample_id":sample[self.id_col]
                    })
            except Exception:
                continue

            cell_data.append(sample_cell_data)
            if len(cell_data) == self.samples_count:
                break

        if len(cell_data) < self.samples_count:
            raise ValueError(f"Only {len(cell_data)} valid samples found, expected {self.samples_count}")

        fig, axes = plt.subplots(
            len(self.dataset_keys),
            self.samples_count,
            figsize=(self.samples_count * 3, len(self.dataset_keys) * 3),
            squeeze=False
        )

        for dataset_idx, dataset_key in enumerate(self.dataset_keys):
            for sample_idx in range(self.samples_count):
                ax = axes[dataset_idx][sample_idx]
                ax.set_xticks([])
                ax.set_yticks([])

                if sample_idx == 0:
                    ax.set_ylabel(dataset_key)

                sample_cell_data = cell_data[sample_idx][dataset_idx]
                ax.imshow(sample_cell_data["image"])
                ax.set_title(f"id: {sample_cell_data['sample_id']}")

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi)
        buf.seek(0)
        plt.close(fig)

        image = Image.open(buf)
        return image