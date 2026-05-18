from typing import List, Any, Dict, Literal

from .dataset_creator import DatasetCreator

class ConcatDatasetCreator(DatasetCreator):
    def __init__(self, *, base_creators: Dict[str, DatasetCreator],
                        dataset_id_field: str,
                        missing_value: Any) -> None:
        self.base_creators = base_creators
        self.dataset_id_field = dataset_id_field
        self.missing_value = missing_value

    def __call__(self) -> Dict[str, List[Any]]:
        base_datasets = {name: creator()  for name, creator in self.base_creators.items()}
        dataset = {
            self.dataset_id_field: []
        }
        for name, base_dataset in base_datasets.items():
            local_keys = list(base_dataset.keys())
            for key in local_keys:
                if key not in dataset:
                    dataset[key] = []
                    for i in range(len(dataset[self.dataset_id_field])):
                        dataset[key].append(self.missing_value)
            for key in dataset.keys():
                values = None
                if key in base_dataset:
                    values = base_dataset[key]
                else:
                    values = [self.missing_value] * len(base_dataset[local_keys[0]])
                dataset[key].extend(values)
            dataset[self.dataset_id_field].extend([name] * len(base_dataset[local_keys[0]]))
        return dataset