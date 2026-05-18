from typing import Any, List, Dict
import json
import os

from .storage import Storage

class JsonStorage(Storage):
    def __init__(self, *, relative_path: str) -> None:
        self.path = None
        self.relative_path = relative_path
    
    def set_path(self, *, path: str) -> None:
        self.path = path
    
    def load(self) -> object:
        assert self.path != None
        final_path = os.path.join(self.path, self.relative_path)
        with open(final_path, "r") as f:
            obj = json.load(f)
        return obj
    
    def save(self, *, obj: object) -> None:
        assert self.path != None
        final_path = os.path.join(self.path, self.relative_path)
        with open(final_path, "w+") as f:
            json.dump(obj, f)