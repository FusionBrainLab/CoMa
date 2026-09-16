import os
import json
from pathlib import Path

from tqdm import tqdm

def main():
    repo_root = os.environ.get(
        "REPO_ROOT",
        str(Path(__file__).resolve().parents[3]),
    )
    results_folder = os.path.join(
        repo_root,
        "experiments/benchmarks/coma/split_evaluation/reproduced_results/upper_results",
    )
    for name in tqdm(os.listdir(results_folder)):
        path = os.path.join(results_folder, name)
        with open(path, "r") as f:
            data = json.load(f)
        submit_name = name.split(".")[0]
        data[submit_name] = data["submit"]
        data.pop("submit")
        with open(path, "w+") as f:
            json.dump(data, f, ensure_ascii=False)

if __name__ == "__main__":
    main()