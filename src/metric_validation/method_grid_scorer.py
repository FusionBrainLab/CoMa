"""Score a materialized method-validation grid with one learned metric instance."""

import json
import os
import re
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..core.base import Function
from ..dataset_creator import InversedJsonChunkDatasetLoader
from ..sample_metric import LearnedEnsembleMetric


RESULT_RE = re.compile(
    r"train-(?P<train_type>.+?)_model-(?P<model_size>[0-9]+)B_"
    r"1d(?P<context_1d_count>[0-9]+)_"
    r"2d(?P<context_2d_count>[0-9]+)_"
    r"3d(?P<context_3d_count>[0-9]+)\.json$"
)


class MethodGridScorer(Function):
    def __init__(self, *, reference_results_root: str,
                        submit_root: str,
                        output_results_root: str,
                        model_path: str,
                        feature_metrics: Dict[str, Any],
                        metric_name: str = "contextual_relevance",
                        sample_massing_key: str = "pred_massing",
                        verbose: bool = True) -> None:
        self.reference_results_root = reference_results_root
        self.submit_root = submit_root
        self.output_results_root = output_results_root
        self.metric_name = metric_name
        self.verbose = verbose
        self.model_path = model_path
        self.feature_metrics = feature_metrics
        self.sample_massing_key = sample_massing_key
        self.sample_metric = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _submit_dir(self, config: Dict[str, str]) -> str:
        return os.path.join(
            self.submit_root,
            config["train_type"],
            "%sB" % config["model_size"],
            "1d%s_2d%s_3d%s" % (
                config["context_1d_count"],
                config["context_2d_count"],
                config["context_3d_count"],
            ),
        )

    def _score_submit(self, submit_dir: str) -> tuple[float, int, int]:
        submit = InversedJsonChunkDatasetLoader(
            folder_path=submit_dir,
            name_pattern=".*",
            verbose=False,
        )()
        rows = pd.DataFrame(submit)
        values = []
        for _, row in tqdm(rows.iterrows(), total=len(rows), disable=not self.verbose):
            try:
                value = float(self.sample_metric(sample=row.to_dict()))
                if np.isfinite(value):
                    values.append(value)
            except Exception:
                continue
        if len(values) == 0:
            raise ValueError("no valid samples in %s" % submit_dir)
        return float(np.mean(values)), len(values), len(rows)

    def __call__(self) -> Dict[str, int]:
        os.makedirs(self.output_results_root, exist_ok=True)
        self._log("building learned metric and context caches ...")
        self.sample_metric = LearnedEnsembleMetric(
            model_path=self.model_path,
            feature_metrics=self.feature_metrics,
            sample_massing_key=self.sample_massing_key,
        )
        self._log("learned metric ready")
        scored = 0
        missing = 0
        failed = 0
        for file_name in sorted(os.listdir(self.reference_results_root)):
            match = RESULT_RE.match(file_name)
            if match is None:
                continue
            submit_name = file_name[:-5]
            submit_dir = self._submit_dir(match.groupdict())
            if not os.path.isdir(submit_dir):
                missing += 1
                self._log("missing submit folder: %s" % submit_dir)
                continue
            try:
                score, valid_count, total_count = self._score_submit(submit_dir)
            except Exception as exc:
                failed += 1
                self._log("failed %s: %s" % (submit_name, exc))
                continue
            output = {submit_name: {self.metric_name: score}}
            output_path = os.path.join(self.output_results_root, file_name)
            with open(output_path, "w") as file:
                json.dump(output, file, ensure_ascii=False)
            scored += 1
            self._log("%s: %.6f (%d/%d valid)" % (submit_name, score, valid_count, total_count))
        return {"scored": scored, "missing": missing, "failed": failed}
