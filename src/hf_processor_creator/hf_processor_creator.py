from abc import ABC, abstractmethod
from typing import List, Any, Dict

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.processing_utils import ProcessorMixin

from ..core.base import Function

class HFProcessorCreator(Function, ABC):
    @abstractmethod
    def __call__(self) -> PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin:
        ...
