from typing import Dict, Any, List

from torch.utils.data import Dataset, DataLoader

class DictDataset(Dataset):
    def __init__(self, *, dict_dataset: Dict[str, List[Any]]):
        self.dict_dataset = dict_dataset
        self.cols = list(dict_dataset.keys())

    def __getitem__(self, index):
        sample = {k: self.dict_dataset[k][index] for k in self.cols}
        return sample

    def __len__(self):
        return len(self.dict_dataset[self.cols[0]])