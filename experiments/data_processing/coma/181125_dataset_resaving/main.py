import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sys
sys.path.append("/workspace-SR008.fs2/maslov/massing_generation")
from src.dataset_creator import InversedJsonDatasetLoader
from src.dataset_handler import InversedJsonDatasetSaver

def main():
    paths = [
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/071125_train_test_split/train_dataset.json",
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/071125_train_test_split/test_dataset.json"
    ]
    new_paths = [
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/181125_dataset_resaving/train_dataset.json",
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/181125_dataset_resaving/test_dataset.json"
    ]
    #Resave images
    new_folders = [
        "env_render_high", "env_render_low", "env_image", "map_image_base", "map_image_alt", "massing_image"
    ]
    old_folders = [
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/261025_images_rendering/env_renders_high",
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/261025_images_rendering/env_renders_low",
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/231025_images_sampling/env_image",
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/231025_images_sampling/map_image_base",
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/231025_images_sampling/map_image_alt",
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/091125_dataset_overview/massing_images"
    ]
    """new_base = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/181125_dataset_resaving"
    for name, old_folder in zip(new_folders, old_folders):
        new_folder = os.path.join(new_base, name)
        os.mkdir(new_folder)
        for filename in tqdm(os.listdir(old_folder)):
            old_path = os.path.join(old_folder, filename)
            new_path = os.path.join(new_folder, filename)
            shutil.copy(old_path, new_path)"""

    #Remap links
    tqdm.pandas()
    for p, new_p in zip(paths, new_paths):
        loader = InversedJsonDatasetLoader(path=p)
        dataset = loader()
        dataset = pd.DataFrame(dataset)
        dataset["id"] = dataset.apply(lambda row: str(row["id"]), axis=1)
        dataset["id_buffer"] = dataset["id"]
        dataset = dataset.set_index("id_buffer")

        dataset["massing_image"] = dataset.progress_apply(lambda row: os.path.join("massing_image", f"{row["id"]}.png"), axis=1)
        for name in new_folders:
            if name == "massing_image":
                continue
            dataset[name] = dataset.progress_apply(lambda row: os.path.join(name, os.path.basename(row[name])), axis=1)

        saver = InversedJsonDatasetSaver(path=new_p)
        saver(dataset=dataset.to_dict("list"))
    
    #Resave property dataset
    property_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/171025_data_pipeline/dataset.json"
    loader = InversedJsonDatasetLoader(path=property_dataset_path)
    property_dataset = loader()
    property_dataset = pd.DataFrame(property_dataset)
    property_dataset["id_buffer"] = property_dataset["id"]
    property_dataset = property_dataset.set_index("id_buffer")

    property_dataset = property_dataset[["id", "massing"]]
    property_dataset = property_dataset.to_dict("list")
    new_property_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/data_processing/181125_dataset_resaving/env_massings.json"
    saver = InversedJsonDatasetSaver(path=new_property_path)
    saver(dataset=property_dataset)

if __name__ == "__main__":
    main()