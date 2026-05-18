from torch import bfloat16

from .core.attr_access import *
from .core.base import *
from .core.event_utils import *
from .core.function_utils import *
from .core.parsers import *

from .dataset_creator import *
from .dataset_processor import *
from .dataset_handler import *
from .metric import *
from .dfs import *
from .interpretable_function import *
from .massing_to_trimesh_converter import *
from .point_geo_projector import *
from .polys_geo_projector import *
from .md_point_converter import *
from .md_polygons_converter import *
from .polygons_to_shapely_converter import *
from .data_loader import *
from .dict_to_content_parser import *
from .dict_to_memory_parser import *
from .dict_to_string_parser import *
from .floats_merger import *
from .fsdp_model_schedule import *
from .gradient_computer import *
from .gradient_norm_computer import *
from .gradient_optimizer import *
from .index_sharder import *
from .logger import *
from .loss import *
from .memory_tokenizer import *
from .metric import *
from .model_creator import *
from .sample_metric import *
from .storage import *
from .torch_state_manager import *
from .training_strategy import *
from .agent_message import *
from .checkpoint_manager import *
from .distributed_hardware_launcher import *
from .torch_nn_module import *
from .local_rank_function import *
from .string_formatter import *
from .batch_sharder import *
from .tensor_sharder import *
from .model_handler import *
from .submit_creator import *
from .agent import *
from .agent_message import *
from .memory_to_dict_parser import *
from .error_safe_function import *
from .memory_to_string_parser import *
from .validator import *
from .string_extractor import *
from .string_deserializer import *
from .memory_indicator import *
from .precompute_function import *
from .vllm_inference import *
from .distributions_visualizer import *
from .retry_function import *
from .image_data_visualizer import *
from .serializer import *
from .massing_cad_executor import *
from .torch_dataset import *
from .data_collator import *
from .hf_processor_creator import *
from .tensor_batch_processor import *
from .generation_strategy import *
from .massing_to_massing_mesh_converter import *
from .mesh_compiler import *
from .langgraph_module import *
from .langgraph_node import *
from .langgraph_condition_function import *
from .dict_creator import *
from .save_visualizer import *
from .dataset_merger import *
from .polygons_analyzer import *
from .sample_processor import *

def load_config(json_config):
    if type(json_config) == dict:
        if "type" in json_config:
            params = {k: load_config(v) for k, v in json_config.items() if k != "type"}
            return globals()[json_config["type"]](**params)
        elif "type_static" in json_config:
            return globals()[json_config["type_static"]]
        else:
            return {k: load_config(v) for k, v in json_config.items()}
    elif type(json_config) == list or type(json_config) == tuple:
        return [load_config(v) for v in json_config]
    else:
        return json_config