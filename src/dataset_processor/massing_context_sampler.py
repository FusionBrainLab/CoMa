from copy import deepcopy
from pathlib import Path
import random
from typing import List, Any, Dict
import os

import pandas as pd
from tqdm import tqdm

from .dataset_processor import DatasetProcessor

class MassingContextSampler(DatasetProcessor):
    def __init__(self, *, multi_image_col_to_folder: Dict[str, str],
                        single_image_col_to_folder: Dict[str, str],
                        context_ids_col: str,
                        json_context_col: str,
                        massing_col: str,
                        id_col: str,
                        context_col_probs: Dict[str, float],
                        no_context_prob: float,
                        multi_image_count_ranges: Dict[str, List[int]],
                        json_count_range: List[int],
                        random_seed: int,
                        n_samples: int,
                        max_context_cols: int) -> None:
        self.multi_image_col_to_folder = multi_image_col_to_folder
        self.single_image_col_to_folder = single_image_col_to_folder
        self.context_ids_col = context_ids_col
        self.json_context_col = json_context_col
        self.massing_col = massing_col
        self.id_col = id_col
        self.context_col_probs = context_col_probs
        self.no_context_prob = no_context_prob
        self.multi_image_count_ranges = multi_image_count_ranges
        self.json_count_range = json_count_range
        self.random_seed = random_seed
        self.n_samples = n_samples
        self.max_context_cols = max_context_cols
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        base_pd_dataset = pd.DataFrame(dataset)
        indexed_pd_dataset = base_pd_dataset.set_index(self.id_col, drop=False)

        context_cols = (
            list(self.single_image_col_to_folder.keys())
            + list(self.multi_image_col_to_folder.keys())
            + [self.json_context_col]
        )

        new_dataset = {k: [] for k in list(dataset.keys()) + [col for col in context_cols if col not in dataset]}
        rng = random.Random(self.random_seed)

        for _, row in tqdm(base_pd_dataset.iterrows(), total=len(base_pd_dataset)):
            for _ in range(self.n_samples):
                for col in new_dataset.keys():
                    if col in context_cols:
                        new_dataset[col].append(None)
                    else:
                        new_dataset[col].append(row[col])

                if rng.random() < self.no_context_prob:
                    continue

                sampled_context_cols = []
                for col in context_cols:
                    if rng.random() < self.context_col_probs.get(col, 0):
                        sampled_context_cols.append(col)

                if len(sampled_context_cols) > self.max_context_cols:
                    sampled_context_cols = rng.sample(sampled_context_cols, self.max_context_cols)
                sampled_context_cols = set(sampled_context_cols)

                for col, folder in self.single_image_col_to_folder.items():
                    if col not in sampled_context_cols:
                        continue

                    sample_id = str(row[self.id_col])
                    image_path = os.path.join(folder, f"{sample_id}.png")
                    new_dataset[col][-1] = image_path

                for col, folder in self.multi_image_col_to_folder.items():
                    if col not in sampled_context_cols:
                        continue

                    sample_id = str(row[self.id_col])
                    base_path = os.path.join(folder, sample_id)
                    image_paths = [os.path.join(base_path, name) for name in os.listdir(base_path) if name.endswith(".png")]
                    count_range = self.multi_image_count_ranges.get(col, [len(image_paths), len(image_paths)])
                    context_count = rng.randint(count_range[0], count_range[1])
                    context_count = min(context_count, len(image_paths))
                    new_dataset[col][-1] = rng.sample(image_paths, context_count) if context_count > 0 else []
                
                if self.json_context_col not in sampled_context_cols:
                    continue

                count_range = self.json_count_range
                context_ids = list(row[self.context_ids_col])
                context_count = rng.randint(count_range[0], count_range[1])
                context_count = min(context_count, len(context_ids))
                sampled_context_ids = rng.sample(context_ids, context_count) if context_count > 0 else []

                context_massings = []
                building_idx = 0
                for context_id in sampled_context_ids:
                    if context_id not in indexed_pd_dataset.index:
                        continue
                    context_massing = indexed_pd_dataset.loc[context_id][self.massing_col]
                    for building in context_massing:
                        context_building = deepcopy(building)
                        context_building["id"] = str(building_idx)
                        context_massings.append(context_building)
                        building_idx += 1
                new_dataset[self.json_context_col][-1] = context_massings

        return new_dataset