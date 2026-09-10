from typing import List, Any, Dict, Tuple
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .dataset_creator import DatasetCreator

def load_chunk(item: Tuple[int, Path, str, str]) -> Tuple[int, Dict[str, List[Any]], List[str]]:
    i, path, feature, path_feature = item
    with open(path, "r") as f:
        data = json.load(f)

    keys = list(data[0].keys())
    chunk = {k: [data[j][k] for j in range(len(data))] for k in keys}
    chunk[path_feature] = [feature] * len(data)
    if path_feature not in keys:
        keys.append(path_feature)

    return i, chunk, keys

class InversedJsonChunkTreeDatasetLoader(DatasetCreator):
    def __init__(self, *, folder_path: str,
                        name_pattern: str,
                        path_feature: str,
                        num_workers: int) -> None:
        self.folder_path = folder_path
        self.name_pattern = name_pattern
        self.path_feature = path_feature
        self.num_workers = num_workers
    
    def __call__(self) -> Dict[str, List[Any]]:
        paths = [p for p in Path(self.folder_path).rglob(self.name_pattern) if p.is_file()]
        features = [str(p.relative_to(self.folder_path).parent) for p in paths]
        items = [
            (i, path, feature, self.path_feature)
            for i, (path, feature) in enumerate(zip(paths, features))
        ]

        if self.num_workers == 1:
            results = (load_chunk(item) for item in tqdm(items, total=len(items)))
        else:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(load_chunk, item) for item in items]
                results_by_index = {}
                for future in tqdm(as_completed(futures), total=len(futures)):
                    i, chunk, keys = future.result()
                    results_by_index[i] = (i, chunk, keys)
                results = (results_by_index[i] for i in range(len(items)))

        dataset = None
        for _, chunk, keys in results:
            if dataset is None:
                dataset = chunk
            else:
                for key in keys:
                    dataset[key].extend(chunk[key])
        return dataset
