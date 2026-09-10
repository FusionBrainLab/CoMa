from copy import deepcopy
from pathlib import Path
import random
from typing import List, Any, Dict, Literal, Optional
import os

import pandas as pd
from tqdm import tqdm

from .dataset_processor import DatasetProcessor
from ..dataset_creator import DatasetCreator

class MassingContextConverter(DatasetProcessor):
    def __init__(self, *, multi_image_col_to_folder: Dict[str, str],
                        single_image_col_to_folder: Dict[str, str],
                        context_ids_col: str,
                        json_context_col: str,
                        massing_col: str,
                        id_col: str,
                        context_type_to_counts: Dict[Literal["json", "single_image", "multi_image"], int],
                        random_seed: int,
                        context_dataset_loader: DatasetCreator,
                        sampled_context_ids_col: Optional[str] = None) -> None:
        self.multi_image_col_to_folder = multi_image_col_to_folder
        self.single_image_col_to_folder = single_image_col_to_folder
        self.context_ids_col = context_ids_col
        self.json_context_col = json_context_col
        self.massing_col = massing_col
        self.id_col = id_col
        self.context_type_to_counts = context_type_to_counts
        self.random_seed = random_seed
        self.sampled_context_ids_col = sampled_context_ids_col
        self.rng = random.Random(self.random_seed)
        context_dataset = context_dataset_loader()
        pd_context_dataset = pd.DataFrame(context_dataset)
        self.indexed_context_dataset = pd_context_dataset.set_index(self.id_col, drop=False)

    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        base_pd_dataset = pd.DataFrame(dataset)
        indexed_pd_dataset = base_pd_dataset.set_index(self.id_col, drop=False)

        context_cols = (
            list(self.single_image_col_to_folder.keys())
            + list(self.multi_image_col_to_folder.keys())
            + [self.json_context_col]
        )

        new_cols = context_cols + ([self.sampled_context_ids_col] if self.sampled_context_ids_col else [])
        new_dataset = {k: [] for k in list(dataset.keys()) + [col for col in new_cols if col not in dataset]}

        for _, row in tqdm(base_pd_dataset.iterrows(), total=len(base_pd_dataset)):
            for col in new_dataset.keys():
                if col in new_cols:
                    new_dataset[col].append(None)
                else:
                    new_dataset[col].append(row[col])

            if "json" in self.context_type_to_counts:
                if self.context_type_to_counts["json"] > 0:
                    context_ids = list(row[self.context_ids_col])
                    sampled_context_ids = self.rng.sample(context_ids, self.context_type_to_counts["json"]) if len(context_ids) > self.context_type_to_counts["json"] else context_ids
                    context_massings = []
                    existing_context_ids = []
                    building_idx = 0
                    for context_id in sampled_context_ids:
                        if context_id not in self.indexed_context_dataset.index:
                            continue
                        existing_context_ids.append(context_id)
                        context_massing = self.indexed_context_dataset.loc[context_id][self.massing_col]
                        for building in context_massing:
                            context_building = deepcopy(building)
                            context_building["id"] = str(building_idx)
                            context_massings.append(context_building)
                            building_idx += 1
                    new_dataset[self.json_context_col][-1] = context_massings
                    if self.sampled_context_ids_col:
                        new_dataset[self.sampled_context_ids_col][-1] = existing_context_ids
            if "single_image" in self.context_type_to_counts:
                if self.context_type_to_counts["single_image"] > 0:
                    for col, folder in self.single_image_col_to_folder.items():
                        sample_id = str(row[self.id_col])
                        image_path = os.path.join(folder, f"{sample_id}.png")
                        new_dataset[col][-1] = image_path
            if "multi_image" in self.context_type_to_counts:
                if self.context_type_to_counts["multi_image"] > 0:
                    for col, folder in self.multi_image_col_to_folder.items():
                        sample_id = str(row[self.id_col])
                        base_path = os.path.join(folder, sample_id)
                        image_paths = [os.path.join(base_path, name) for name in os.listdir(base_path) if name.endswith(".png")]
                        context = self.rng.sample(image_paths, self.context_type_to_counts["multi_image"]) if len(image_paths) > self.context_type_to_counts["multi_image"] else image_paths
                        new_dataset[col][-1] = context
        return new_dataset