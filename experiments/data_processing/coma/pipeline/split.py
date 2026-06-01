import json
import math
import os
from copy import deepcopy

import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, Polygon, MultiPolygon
import shapely
import trimesh
from pandarallel import pandarallel
from diffusers import (
        QwenImageEditPipeline,
        QwenImageEditPlusPipeline
    )
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

import sys
sys.path.append("/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation")
from src.interpretable_function import InterpretableFunction
from src.core.function_utils import FunctionWrapper, FunctionGraph
from src.core.parsers import (
    DictToDictParser, AnyToDictParser, DictToAnyParser
)
from src.dataset_creator import (
    CsvDatasetCreator,
    PreprocessDatasetCreator,
    CoMaRegionsCreator,
    CoMaBuildingsCreator,
    InversedJsonChunkDatasetLoader
)
from src.dataset_processor import (
    GeoJSONToShapelyProcessor,
    GeoProjectProcessor,
    CompositionProcessor,
    FunctionProcessor,
    FootprintsToBuildingProcessor
)
from src.dataset_merger import GeoPandasDatasetIntersector, CoMaBuildingsRegionsMetadataMerger
from src.dataset_handler import InversedJsonChunkDatasetSaver
from src.mesh_compiler import MassingMeshCompiler, PolygonMeshCompiler
from src.image_data_visualizer import FocusMeshVisualizer, MapRegionVisualizer

def main():
    pandarallel.initialize(progress_bar=True, nb_workers=5)
    tqdm.pandas()
    loader = InversedJsonChunkDatasetLoader(
        folder_path="/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/data_processing/coma/dataset/massings",
        name_pattern=".*",
        verbose=True
    )
    massings = loader()
    pd_massings = pd.DataFrame(massings)
    train_massings, test_massings = train_test_split(pd_massings, test_size=0.2, random_state=42)
    train_massings = train_massings.to_dict("list")
    test_massings = test_massings.to_dict("list")
    train_saver = InversedJsonChunkDatasetSaver(
        folder_path="/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/data_processing/coma/splits/train",
        chunk_length=10000
    )
    test_saver = InversedJsonChunkDatasetSaver(
        folder_path="/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/data_processing/coma/splits/test",
        chunk_length=10000
    )
    train_saver(dataset=train_massings)
    test_saver(dataset=test_massings)

if __name__ == "__main__":
    main()