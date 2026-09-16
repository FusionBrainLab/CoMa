import json
import math
import os
from copy import deepcopy
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, Polygon, MultiPolygon
from pandarallel import pandarallel
from PIL import Image
from sklearn.model_selection import train_test_split

import sys
REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    str(Path(__file__).resolve().parents[4]),
)
sys.path.append(REPO_ROOT)
from src.interpretable_function import InterpretableFunction
from src.core.function_utils import FunctionWrapper, FunctionGraph
from src.core.parsers import (
    DictToDictParser, AnyToDictParser, DictToAnyParser
)
from src.dataset_creator import (
    InversedJsonChunkDatasetLoader
)
from src.dataset_handler import InversedJsonChunkDatasetSaver

def main():
    pandarallel.initialize(progress_bar=True, nb_workers=5)
    tqdm.pandas()
    loader = InversedJsonChunkDatasetLoader(
        folder_path=os.path.join(REPO_ROOT, "experiments/data_processing/coma/dataset/massings"),
        name_pattern=".*",
        verbose=True
    )
    massings = loader()
    pd_massings = pd.DataFrame(massings)
    print(len(pd_massings))
    train_massings, test_massings = train_test_split(pd_massings, test_size=0.2, random_state=42)
    print(len(train_massings))
    print(len(test_massings))

    sample = pd_massings.iloc[0].to_dict()
    with open(os.path.join(REPO_ROOT, "experiments/data_processing/coma/pipeline/sample.json"), "w") as f:
        json.dump(sample, f)
    
    return
    train_massings = train_massings.to_dict("list")
    test_massings = test_massings.to_dict("list")
    train_saver = InversedJsonChunkDatasetSaver(
        folder_path=os.path.join(REPO_ROOT, "experiments/data_processing/coma/splits/train"),
        chunk_length=10000
    )
    test_saver = InversedJsonChunkDatasetSaver(
        folder_path=os.path.join(REPO_ROOT, "experiments/data_processing/coma/splits/test"),
        chunk_length=10000
    )
    train_saver(dataset=train_massings)
    test_saver(dataset=test_massings)

if __name__ == "__main__":
    main()