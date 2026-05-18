from .fsdp_model_schedule import *

class ConstantSchedule(FSDPModelSchedule):
    const: bool
    def __init__(self, *, const: bool) -> None:
        self.const = const
    def __call__(self, *, forward: int) -> bool:
        return self.const