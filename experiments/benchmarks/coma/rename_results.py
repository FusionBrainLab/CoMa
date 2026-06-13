import os
import json

from tqdm import tqdm

def main():
    results_folder = "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results"
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