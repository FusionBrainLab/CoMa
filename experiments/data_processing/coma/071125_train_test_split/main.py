import os

import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append("/workspace-SR008.fs2/maslov/massing_generation")
from src.dataset_creator import InversedJsonDatasetLoader
from src.dataset_handler import InversedJsonDatasetSaver

def main():
    loader = InversedJsonDatasetLoader(path="/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/271025_final_data_processing/dataset.json")
    dataset = loader()
    dataset = pd.DataFrame(dataset)
    dataset["id"] = dataset.apply(lambda row: str(row["id"]), axis=1)
    dataset["id_buffer"] = dataset["id"]
    dataset = dataset.set_index("id_buffer")

    splits = train_test_split(dataset, test_size=0.1, random_state=42)
    names = ["train_dataset", "test_dataset"]
    folder = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/071125_train_test_split"
    
    for name, split in zip(names, splits):
        split_dataset = split.to_dict("list")
        path = os.path.join(folder, f"{name}.json")
        saver = InversedJsonDatasetSaver(path=path)
        saver(dataset=split_dataset)

if __name__ == "__main__":
    main()