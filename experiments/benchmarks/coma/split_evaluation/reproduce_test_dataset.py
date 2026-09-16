import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":
    dims = {
        "context_1d_count": [0, 2, 4, 6, 8, 10],
        "context_2d_count": [0, 1],
        "context_3d_count": [0, 2, 4, 6, 8],
    }

    runs = [{
        "train_type": "no_context",
        "context_1d_count": 0,
        "context_2d_count": 0,
        "context_3d_count": 0,
    }, {
        "train_type": "unimodal_context",
        "context_1d_count": 0,
        "context_2d_count": 0,
        "context_3d_count": 0,
    }]

    for dim, values in dims.items():
        zero_dims = {key: 0 for key in dims.keys() if key != dim}
        for value in values:
            if value == 0:
                continue
            runs.append({
                **zero_dims,
                dim: value,
                "train_type": "unimodal_context",
            })

    for context_1d_count in dims["context_1d_count"]:
        for context_2d_count in dims["context_2d_count"]:
            for context_3d_count in dims["context_3d_count"]:
                runs.append({
                    "train_type": "multimodal_context",
                    "context_1d_count": context_1d_count,
                    "context_2d_count": context_2d_count,
                    "context_3d_count": context_3d_count,
                })

    base_path = os.environ.get(
        "REPO_ROOT",
        str(Path(__file__).resolve().parents[4]),
    )
    config_path = (
        f"{base_path}/experiments/benchmarks/coma/split_evaluation/"
        "reproduce_test_dataset.yaml"
    )
    env = {**os.environ, "HYDRA_FULL_ERROR": "1", "REPO_ROOT": base_path}
    for run in tqdm(runs):
        for model_size in [2, 4, 8]:
            args = [
                f"{name}={value}"
                for name, value in {**run, "model_size": model_size}.items()
            ]
            subprocess.run(
                [
                    sys.executable,
                    f"{base_path}/hydra_run.py",
                    "--config_path",
                    config_path,
                    *args,
                ],
                check=True,
                env=env,
            )
