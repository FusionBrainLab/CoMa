import json
import math
import os
import random
import io
from collections import deque
from copy import copy
import asyncio
from functools import partial
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, Polygon, MultiPolygon
import trimesh
import pyvista as pv
from PIL import Image
from pathos.multiprocessing import ProcessingPool
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("/workspace-SR008.fs2/maslov/massing_generation")
from src.dataset_handler import InversedJsonDatasetSaver
from src.dataset_creator import InversedJsonDatasetLoader, PreprocessDatasetCreator
from src.dataset_processor import (
    ContextualMassingVisualizer,
    FunctionProcessor
)
from src.core.function_utils import (
    FunctionWrapper
)
from src.interpretable_function import InterpretableFunction

def main():
    paths = [
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/training/071125_system_prompt_train/2B/logs/logs.json",
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/training/071125_system_prompt_train/4B/logs/logs.json",
        "/workspace-SR008.fs2/maslov/massing_generation/experiments/training/071125_system_prompt_train/8B/logs/logs.json"
    ]

    losses = []
    for path in paths:
        with open(path, "r") as f:
            local_losses = json.load(f)
        local_losses = [l["loss"] for l in local_losses]
        losses.append(local_losses)

    models = ["CoMa-2B", "CoMa-4B", "CoMa-8B"]

    # Generate sample loss data for multiple models
    data = []
    for i, model in enumerate(models):
        for epoch in range(len(losses[0])):
            data.append({'Model': model, 'Epoch': epoch, 'Loss': losses[i][epoch]})

    # Create DataFrame
    df = pd.DataFrame(data)

    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x='Epoch', y='Loss', hue='Model', style='Model', 
                palette='viridis', markers=False, dashes=True)

    local_fontsize = 18
    plt.xlabel('Step', fontsize=local_fontsize)
    plt.ylabel('Loss', fontsize=local_fontsize)
    ax.tick_params(axis='x', labelsize=local_fontsize)
    ax.tick_params(axis='y', labelsize=local_fontsize)

    plt.grid(True, alpha=0.2)
    plt.legend(title='Models', fontsize=local_fontsize)
    legend = ax.get_legend()
    plt.setp(legend.get_title(), fontsize=local_fontsize)

    plt.tight_layout()
    plt.savefig("/workspace-SR008.fs2/maslov/massing_generation/experiments/training/071125_system_prompt_train/visualization/loss.png")

if __name__ == "__main__":
    main()