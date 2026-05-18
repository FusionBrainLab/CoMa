from typing import List, Any, Dict
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from tqdm import tqdm
import mmap
import orjson

from .dataset_creator import DatasetCreator

def load_chunk(i, path):
    with open(path, "r") as f:
        data = json.load(f)
    keys = list(data[0].keys())
    chunk = {k: [data[i][k] for i in range(len(data))] for k in keys}
    return {i: chunk}
class InversedJsonChunkDatasetLoader(DatasetCreator):
    def __init__(self, *, folder_path: str,
                        name_pattern: str,
                        verbose: bool) -> None:
        self.folder_path = folder_path
        self.name_pattern = name_pattern
        self.verbose = verbose
    
    def __call__(self) -> Dict[str, List[Any]]:
        names = os.listdir(self.folder_path)
        valid_names = [n for n in names if re.fullmatch(self.name_pattern, n)]

        dataset = None
        cycle = tqdm(valid_names) if self.verbose else valid_names
        for name in cycle:
            path = os.path.join(self.folder_path, name)

            """data = b''
            with open(path, 'rb') as f:
                while chunk := f.read(256*1024*1024):
                    data += chunk
            data = orjson.loads(data)"""
            """with open(path, 'rb', buffering=2048*1024) as f:
                data = orjson.loads(f.read())"""
            """with open(path, 'r+b') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    data = orjson.loads(memoryview(mmapped_file))"""
            """with open(path, 'rb') as f:
                data = orjson.loads(f.read())"""
            with open(path, "r") as f:
                data = json.load(f)

            keys = list(data[0].keys())
            chunk = {k: [data[i][k] for i in range(len(data))] for k in keys}
            if dataset == None:
                dataset = chunk
            else:
                for k in keys:
                    dataset[k].extend(chunk[k])
        
        """paths = [os.path.join(self.folder_path, name) for name in valid_names]

        results = {}
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_item = {executor.submit(load_chunk, i, p): p for i, p in enumerate(paths)}
            for future in tqdm(as_completed(future_to_item), total=len(paths)):
                result = future.result()
                results[result[0]] = result[1]
        dataset = None
        for i in range(len(paths)):
            chunk = results[i]
            keys = list(chunk.keys())
            if dataset == None:
                dataset = chunk
            else:
                for k in keys:
                    dataset[k].extend(chunk[k])"""

        return dataset