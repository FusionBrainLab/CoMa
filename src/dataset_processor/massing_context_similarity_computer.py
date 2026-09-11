from collections import Counter
from itertools import combinations
import math
from pathlib import Path
from statistics import median
from typing import List, Any, Dict

from tqdm import tqdm

from .dataset_processor import DatasetProcessor
from ..dataset_creator import DatasetCreator

class MassingContextSimilarityComputer(DatasetProcessor):
    def __init__(self, *, train_dataset_loader: DatasetCreator,
                        train_context_ids_col: str,
                        test_context_ids_col: str,
                        train_type_col: str,
                        test_path_col: str,
                        individual_counts_col: str,
                        cooccurrence_counts_col: str,
                        individual_tau_col: str,
                        cooccurrence_tau_col: str,
                        individual_exposure_col: str,
                        coexposure_col: str,
                        individual_coverage_col: str,
                        cooccurrence_coverage_col: str) -> None:
        train_dataset = train_dataset_loader()
        self.test_context_ids_col = test_context_ids_col
        self.test_path_col = test_path_col
        self.individual_counts_col = individual_counts_col
        self.cooccurrence_counts_col = cooccurrence_counts_col
        self.individual_tau_col = individual_tau_col
        self.cooccurrence_tau_col = cooccurrence_tau_col
        self.individual_exposure_col = individual_exposure_col
        self.coexposure_col = coexposure_col
        self.individual_coverage_col = individual_coverage_col
        self.cooccurrence_coverage_col = cooccurrence_coverage_col

        self.train_type_to_individual_counts = {}
        self.train_type_to_cooccurrence_counts = {}
        self.train_type_to_individual_tau = {}
        self.train_type_to_cooccurrence_tau = {}
        for train_type in set(train_dataset[train_type_col]):
            individual_counts = Counter()
            cooccurrence_counts = Counter()
            for local_train_type, context_ids in tqdm(zip(
                train_dataset[train_type_col],
                train_dataset[train_context_ids_col]
            )):
                if local_train_type != train_type or not isinstance(context_ids, list):
                    continue
                unique_context_ids = list(dict.fromkeys(context_ids))
                individual_counts.update(unique_context_ids)
                cooccurrence_counts.update(
                    combinations(sorted(unique_context_ids), 2)
                )
            self.train_type_to_individual_counts[train_type] = individual_counts
            self.train_type_to_cooccurrence_counts[train_type] = cooccurrence_counts
            self.train_type_to_individual_tau[train_type] = (
                median(individual_counts.values())
                if len(individual_counts) > 0
                else 1
            )
            self.train_type_to_cooccurrence_tau[train_type] = (
                median(cooccurrence_counts.values())
                if len(cooccurrence_counts) > 0
                else 1
            )

    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        new_dataset = {key: list(values) for key, values in dataset.items()}
        output_cols = [
            self.individual_counts_col,
            self.cooccurrence_counts_col,
            self.individual_tau_col,
            self.cooccurrence_tau_col,
            self.individual_exposure_col,
            self.coexposure_col,
            self.individual_coverage_col,
            self.cooccurrence_coverage_col,
        ]
        for col in output_cols:
            new_dataset[col] = []

        for context_ids, path in zip(
            dataset[self.test_context_ids_col],
            dataset[self.test_path_col]
        ):
            train_type = Path(path).parts[0]
            individual_counter = self.train_type_to_individual_counts.get(
                train_type,
                Counter()
            )
            cooccurrence_counter = self.train_type_to_cooccurrence_counts.get(
                train_type,
                Counter()
            )
            individual_tau = self.train_type_to_individual_tau.get(train_type, 1)
            cooccurrence_tau = self.train_type_to_cooccurrence_tau.get(train_type, 1)
            unique_context_ids = (
                list(dict.fromkeys(context_ids))
                if isinstance(context_ids, list)
                else []
            )
            pairs = list(combinations(sorted(unique_context_ids), 2))
            individual_counts = [
                individual_counter[context_id]
                for context_id in unique_context_ids
            ]
            cooccurrence_counts = [
                cooccurrence_counter[pair]
                for pair in pairs
            ]

            new_dataset[self.individual_counts_col].append(individual_counts)
            new_dataset[self.cooccurrence_counts_col].append([
                {"ids": list(pair), "count": count}
                for pair, count in zip(pairs, cooccurrence_counts)
            ])
            new_dataset[self.individual_tau_col].append(individual_tau)
            new_dataset[self.cooccurrence_tau_col].append(cooccurrence_tau)
            new_dataset[self.individual_exposure_col].append(
                sum(
                    1 - math.exp(-count / individual_tau)
                    for count in individual_counts
                ) / len(individual_counts)
                if len(individual_counts) > 0
                else 0
            )
            new_dataset[self.coexposure_col].append(
                sum(
                    1 - math.exp(-count / cooccurrence_tau)
                    for count in cooccurrence_counts
                ) / len(cooccurrence_counts)
                if len(cooccurrence_counts) > 0
                else 0
            )
            new_dataset[self.individual_coverage_col].append(
                sum(count > 0 for count in individual_counts)
                / len(individual_counts)
                if len(individual_counts) > 0
                else 0
            )
            new_dataset[self.cooccurrence_coverage_col].append(
                sum(count > 0 for count in cooccurrence_counts)
                / len(cooccurrence_counts)
                if len(cooccurrence_counts) > 0
                else 0
            )

        return new_dataset
