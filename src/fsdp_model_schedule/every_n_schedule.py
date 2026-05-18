from .fsdp_model_schedule import FSDPModelSchedule

class EveryNSchedule(FSDPModelSchedule):
    n: int
    def __init__(self, *, n: int) -> None:
        self.n = n
    def __call__(self, *, forward: int) -> bool:
        return forward % self.n == 0