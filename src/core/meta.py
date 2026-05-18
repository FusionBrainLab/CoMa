import inspect
from typing import Any, get_type_hints, get_args, get_origin
from abc import ABCMeta, ABC

from .design_rules import (
    ArgumentKindsValidation,
    TypingExistanceValidation,
    DefaultsExistanceValidation,
    AbstractionValidation,
    AtomizationValidation,
    InheritanceValidation
)
from .initialization_utils import (
    InheritanceInitialization,
    TypecheckingInitialization
)

class Meta(ABCMeta):
    def __new__(cls, name, bases, namespace):
        validations = [
            ArgumentKindsValidation(),
            TypingExistanceValidation(),
            DefaultsExistanceValidation(),
            AbstractionValidation(),
            AtomizationValidation(),
            InheritanceValidation()
        ]
        initializations = [
            InheritanceInitialization(),
            TypecheckingInitialization()
        ]
        """for val in validations:
            val(name, bases, namespace)"""
        final_namespace = namespace
        for init in initializations:
            final_namespace = init(name, bases, final_namespace)
        
        return super().__new__(cls, name, bases, final_namespace)
