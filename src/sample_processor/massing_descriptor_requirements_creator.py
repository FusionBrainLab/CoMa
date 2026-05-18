from abc import ABC, abstractmethod
from typing import Any, Dict, List
from copy import deepcopy

from .sample_processor import SampleProcessor
from ..sample_metric import SampleMetric

class MassingDescriptorRequirementsCreator(SampleProcessor):
    def __init__(self, *, massing_key: str,
                        base_requirements_key: str,
                        descriptor_requirements: Dict[str, SampleMetric],
                        constant_requirements: Dict[str, Any]) -> None:
        self.massing_key = massing_key
        self.base_requirements_key = base_requirements_key
        self.descriptor_requirements = descriptor_requirements
        self.constant_requirements = constant_requirements

    def __call__(self, *, sample: Dict[str, Any]) -> Dict[str, Any]:
        massing = sample[self.massing_key]
        base_requirements = sample[self.base_requirements_key]
        new_requirements = []
        for i, r in enumerate(base_requirements):
            new_r = deepcopy(r)
            for d in self.descriptor_requirements:
                new_r[d] = self.descriptor_requirements[d](sample={self.massing_key:massing[i]})
            for c in self.constant_requirements:
                new_r[c] = self.constant_requirements[c]
            new_requirements.append(new_r)
        sample.update({self.base_requirements_key:new_requirements})
        return sample