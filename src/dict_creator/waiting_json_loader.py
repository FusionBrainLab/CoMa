from abc import ABC, abstractmethod
from typing import List, Any, Dict
import json
import os
import time

from .dict_creator import DictCreator

class WaitingJsonLoader(DictCreator):
    def __init__(self, *, path: str) -> None:
        self.path = path

    def __call__(self) -> Dict[str, Any]:
        loaded = False
        while not loaded:
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    data = json.load(f)
                loaded = True
            time.sleep(1)
        return data