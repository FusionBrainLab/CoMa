from abc import ABC, abstractmethod
from typing import Any, Dict

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.processing_utils import ProcessorMixin
from transformers import (
    AutoProcessor,
    AutoTokenizer
)

from .hf_processor_creator import HFProcessorCreator

class PretrainedProcessorCreator(HFProcessorCreator):
    def __init__(self, *, cls: str,
                        args: Dict[str, Any]) -> None:
        self.cls = cls
        self.args = args

    def __call__(self) -> PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin:
        processor = globals()[self.cls].from_pretrained(**self.args)
        return processor