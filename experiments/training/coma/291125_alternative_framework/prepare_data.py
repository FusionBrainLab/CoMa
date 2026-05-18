import json

import pandas as pd

import sys
sys.path.append("/workspace-SR008.fs2/maslov/massing_generation")
from src.config_utils import load_config

def main():
    dataset_config_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/training/291125_alternative_framework/dataset_config.json"
    with open(dataset_config_path, "r") as f:
        config = json.load(f)
    dataset_creator = load_config(config)
    dataset = dataset_creator()
    dataset = pd.DataFrame(dataset)

    def process_messages(row):
        messages = {
            "image":[],
            "conversations":[]
        }
        memory = row["memories"]
        role_mapping = {
            "system":"system",
            "user":"human",
            "assistant":"gpt"
        }
        for message in memory:
            value = ""
            for content in message.content:
                if content.modality == "text":
                    value = value + content.content
                elif content.modality == "image":
                    value = value + "<image>"
                    messages["image"].append(content.content)
            new_message = {
                "from":role_mapping[message.role],
                "value":value
            }
            messages["conversations"].append(new_message)
        return messages
    dataset["messages"] = dataset.apply(lambda row: process_messages(row), axis=1)

    new_dataset = dataset["messages"].values.tolist()
    new_dataset_path = "/workspace-SR008.fs2/maslov/massing_generation/experiments/training/291125_alternative_framework/dataset.json"
    with open(new_dataset_path, "w+") as f:
        json.dump(new_dataset, f, ensure_ascii=False)

if __name__ == "__main__":
    main()