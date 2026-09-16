import os
from pathlib import Path

import pandas as pd

from src.dataset_creator import InversedJsonChunkTreeDatasetLoader
from src.dataset_handler import InversedJsonChunkTreeDatasetSaver


if __name__ == "__main__":
    repo_root = os.environ.get(
        "REPO_ROOT",
        str(Path(__file__).resolve().parents[4]),
    )
    base_path = os.path.join(
        repo_root,
        "experiments/benchmarks/coma/split_evaluation/reproduced_dataset",
    )
    path_feature = "inference_path"
    dataset = InversedJsonChunkTreeDatasetLoader(
        folder_path=os.path.join(base_path, "computed_test"),
        name_pattern="*.json",
        path_feature=path_feature,
        num_workers=32,
    )()
    df = pd.DataFrame(dataset)

    lower_dfs = []
    upper_dfs = []
    for path, path_df in df.groupby(path_feature, sort=False):
        if "/1d0_" in path:
            lower_dfs.append(path_df)
            upper_dfs.append(path_df)
            continue
        count = max(1, int(len(path_df) * 0.2))
        lower_dfs.append(path_df.nsmallest(count, "context_coexposure"))
        upper_dfs.append(path_df.nlargest(count, "context_coexposure"))

    for name, dfs in [("lower_20", lower_dfs), ("upper_20", upper_dfs)]:
        InversedJsonChunkTreeDatasetSaver(
            folder_path=os.path.join(base_path, "filtered_test", name),
            chunk_length=10000,
            path_feature=path_feature,
        )(dataset=pd.concat(dfs).to_dict("list"))
