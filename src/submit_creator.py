from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from functools import partial
import math

from tqdm import tqdm

from .core.base import Function
from .dataset_creator import DatasetCreator
from .dataset_handler import DatasetHandler

def execute(row):
    global method
    output = method(**row)
    extended_output = deepcopy(row)
    extended_output.update(output)
    return extended_output

class SubmitCreator(Function):
    def __init__(self, *, dataset_loader: DatasetCreator,
                        method: Function,
                        checkpoint_loader: DatasetCreator | None,
                        submit_saver: DatasetHandler,
                        max_workers: int,
                        batch_size: int) -> None:
        self.dataset_loader = dataset_loader
        self.method = method
        self.checkpoint_loader = checkpoint_loader
        self.submit_saver = submit_saver
        self.max_workers = max_workers
        self.batch_size = batch_size

    def __call__(self) -> None:
        test_dataset = self.dataset_loader()
        keys = list(test_dataset.keys())
        submit = {}
        if self.checkpoint_loader != None:
            submit = self.checkpoint_loader()
            test_dataset = {k: test_dataset[k][len(submit[k]):] for k in keys}
        length = len(test_dataset[keys[0]])
        print(length)

        """def init_worker(instance):
            global method
            method = instance

        max_workers = os.cpu_count() if self.max_workers == -1 else self.max_workers
        rows = [{k: test_dataset[k][i] for k in keys} for i in range(length)]
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(self.method,)) as executor:
            future_to_item = {executor.submit(execute, row): row for row in rows}
            for future in tqdm(as_completed(future_to_item), total=length):
                extended_output = future.result()
                print("Finish")
                for k, v in extended_output.items():
                    if k not in submit:
                        submit[k] = []
                    submit[k].append(v)
                self.submit_saver(dataset=submit)"""

        if self.batch_size == 0:
            for i in tqdm(range(length)):
                row = {k: test_dataset[k][i] for k in keys}
                output = self.method(**row)
                extended_output = deepcopy(row)
                extended_output.update(output)
                for k, v in extended_output.items():
                    if k not in submit:
                        submit[k] = []
                    submit[k].append(v)
                self.submit_saver(dataset=submit)
        else:
            n_batches = math.ceil(length / self.batch_size)
            batch_inds = [[j for j in range(i * self.batch_size, min((i + 1) * self.batch_size, length))] for i in range(n_batches)]
            batches = [{k: [test_dataset[k][ind] for ind in binds] for k in test_dataset.keys()} for binds in batch_inds]
            for i, row in tqdm(enumerate(batches), total=len(batches)):
                output = self.method(**row)
                extended_output = deepcopy(row)
                extended_output.update(output)
                for k, v in extended_output.items():
                    if k not in submit:
                        submit[k] = []
                    submit[k].extend(v)
                self.submit_saver(dataset=submit)
