from typing import List, Any, Dict

import pandas as pd
from tqdm import tqdm
import numpy as np

from .dataset_processor import DatasetProcessor

class FeaturesUniformizeProcessor(DatasetProcessor):
    def __init__(self, *, features: List[str],
                        bins_count: int,
                        min_samples_to_bin: int,
                        min_bin_samples_to_left: int,
                        min_output_samples: int,
                        feature_sets_col: str) -> None:
        self.features = features
        self.bins_count = bins_count
        self.min_samples_to_bin = min_samples_to_bin
        self.min_bin_samples_to_left = min_bin_samples_to_left
        self.min_output_samples = min_output_samples
        self.feature_sets_col = feature_sets_col
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pd_dataset = pd.DataFrame(dataset)
        base_cols = pd_dataset.columns.tolist()

        #----------SPLITTING----------
        BINS_COUNT = self.bins_count
        MIN_BIN_SAMPLES = self.min_samples_to_bin
        MIN_TRAIN_BIN_SAMPLES = self.min_bin_samples_to_left
        MIN_VAL_SAMPLES = self.min_output_samples

        #----------CREATE BINS----------

        features = self.features
        features_bins = {f:[] for f in features}
        for feature in tqdm(features):
            values = pd_dataset[feature].values.tolist()
            base_bounds = np.linspace(min(values), max(values), num=BINS_COUNT, endpoint=True).tolist()
            bin_sample_counts = []
            for i in range(1, len(base_bounds)):
                local_count = len(pd_dataset[(pd_dataset[feature] >= base_bounds[i - 1]) & (pd_dataset[feature] < base_bounds[i])])
                bin_sample_counts.append(local_count)
            final_bin_inds = [[i] for i in range(len(bin_sample_counts))]
            while not all([sum([bin_sample_counts[i] for i in local_inds]) >= MIN_BIN_SAMPLES for local_inds in final_bin_inds]):
                if len(final_bin_inds) == 1:
                    break
                for j in range(len(final_bin_inds)):
                    if sum([bin_sample_counts[i] for i in final_bin_inds[j]]) < MIN_BIN_SAMPLES:
                        if j == 0:
                            final_bin_inds[1].append(j)
                            final_bin_inds.pop(0)
                        elif j == len(final_bin_inds) - 1:
                            final_bin_inds[-1].append(j)
                            final_bin_inds.pop(-1) 
                        else:
                            left_sum = sum([bin_sample_counts[k] for k in final_bin_inds[j - 1]])
                            right_sum = sum([bin_sample_counts[k] for k in final_bin_inds[j + 1]])
                            if left_sum < right_sum:
                                final_bin_inds[j - 1].append(j)
                                final_bin_inds.pop(j - 1) 
                            else:
                                final_bin_inds[j + 1].append(j)
                                final_bin_inds.pop(j + 1) 
                        break
            for inds in final_bin_inds:
                left = base_bounds[min(inds)]
                right = base_bounds[max(inds) + 1]
                features_bins[feature].append([left, right])
        def get_bin(row, bins, feature):
            local_bin = None
            for i, b in enumerate(bins[feature]):
                if row[feature] >= b[0] and row[feature] < b[1]:
                    local_bin = i
                    break
            return local_bin
        def get_bins_vectorized(values, bins):
            # Get unique edges (left edge of first bin + right edges of all bins)
            left_edges = np.array([b[0] for b in bins])
            right_edges = np.array([b[1] for b in bins])
            
            # Create full edge array: first left edge + all right edges
            edges = np.concatenate([[left_edges[0]], right_edges])
            
            # digitize returns 1-based indices, so subtract 1
            bin_indices = np.digitize(values, edges) - 1
            
            # Clip to valid range (0 to len(bins)-1)
            bin_indices = np.clip(bin_indices, 0, len(bins) - 1)
            
            return bin_indices
        for f in tqdm(features):
            pd_dataset[f"bin_{f}"] = get_bins_vectorized(pd_dataset[f].values, features_bins[f])
            #dataset[f"bin_{f}"] = dataset.progress_apply(lambda row: get_bin(row, features_bins, f), axis=1)
        
        #----------OPTIMIZE----------
        VARIATIONAL_BINS_COUNT = 100
        variational_features_bins = {f:[] for f in features}
        for f in features:
            values = pd_dataset[f].values.tolist()
            space = np.linspace(min(values), max(values), num=VARIATIONAL_BINS_COUNT, endpoint=True).tolist()
            bins = [[space[i - 1], space[i]] for i in range(1, len(space))]
            variational_features_bins[f] = bins
        for f in tqdm(features):
            pd_dataset[f"bin_var_{f}"] = get_bins_vectorized(pd_dataset[f].values, variational_features_bins[f])
            #dataset[f"bin_var_{f}"] = dataset.progress_apply(lambda row: get_bin(row, variational_features_bins, f), axis=1)
        
        pd_dataset["id"] = pd_dataset.apply(lambda row: str(row["id"]), axis=1)
        pd_dataset["id_buffer"] = pd_dataset["id"]
        pd_dataset = pd_dataset.set_index("id_buffer")

        feature_val_sets = {f:[] for f in features}
        for f in tqdm(features):
            stop = False
            cur_val_set = []
            cur_train_set = pd.DataFrame(pd_dataset)
            while not stop:
                bins_covered = []
                def get_local_set(row):
                    local_bin = row[f"bin_var_{f}"]
                    if local_bin not in bins_covered:
                        bins_covered.append(local_bin)
                        return True
                    else:
                        return False 
                local_set = cur_train_set[cur_train_set.apply(get_local_set, axis=1)]
                local_train_set = cur_train_set.loc[[i for i in cur_train_set["id"].values.tolist() if i not in local_set["id"].values.tolist()]]

                min_val_match = len(cur_val_set) + len(local_set) >= MIN_VAL_SAMPLES
                min_train_match = all([len(local_train_set[local_train_set[f"bin_{f}"] == i]) < MIN_TRAIN_BIN_SAMPLES for i in range(len(features_bins[f]))])
                if min_val_match or min_train_match:
                    stop = True
                else:
                    if len(cur_val_set) == 0:
                        cur_val_set = local_set
                    else:
                        cur_val_set = pd.concat([cur_val_set, local_set])
                    cur_train_set = local_train_set
            feature_val_sets[f] = cur_val_set
        id_to_set = {}
        def get_set(row, f):
            local_id = row["id"]
            if local_id not in id_to_set:
                id_to_set[local_id] = [f]
            else:
                id_to_set[local_id].append(f)
        for f, local_set in feature_val_sets.items():
            local_set.apply(lambda row: get_set(row, f), axis=1)
        
        total_val_set_ids = set()
        for f, local_set in feature_val_sets.items():
            total_val_set_ids = total_val_set_ids.union(set(local_set["id"].values.tolist()))
        val_set = pd_dataset.loc[list(total_val_set_ids)]
        val_set[self.feature_sets_col] = val_set.apply(lambda row: id_to_set[row["id"]], axis=1)
        val_set = val_set[base_cols + [self.feature_sets_col]]
        
        output_dataset = val_set.to_dict("list")
        return output_dataset