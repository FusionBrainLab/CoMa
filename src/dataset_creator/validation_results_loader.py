from typing import List, Any, Dict
import os
import json

from .dataset_creator import DatasetCreator
from ..string_deserializer import StringDeserializer

class ValidationResultsLoader(DatasetCreator):
    def __init__(self, *, results_folder: str,
                        submit_name_to_dict_deserializer: StringDeserializer,
                        deserialize_filename: bool = False) -> None:
        self.results_folder = results_folder
        self.submit_name_to_dict_deserializer = submit_name_to_dict_deserializer
        self.deserialize_filename = deserialize_filename

    def __call__(self) -> Dict[str, List[Any]]:
        results = []
        for name in os.listdir(self.results_folder):
            with open(os.path.join(self.results_folder, name), "r") as f:
                results.append((name, json.load(f)))
        rows = []
        for name, result in results:
            for submit_name, value_dict in result.items():
                if self.deserialize_filename:
                    submit_name = os.path.splitext(name)[0]
                submit_dict = self.submit_name_to_dict_deserializer(string=submit_name)
                assert isinstance(submit_dict, dict)
                row = {**submit_dict, **value_dict}
                rows.append(row)
        dataset = {}
        for row in rows:
            for key, value in row.items():
                if key not in dataset:
                    dataset[key] = []
                dataset[key].append(value)
        return dataset