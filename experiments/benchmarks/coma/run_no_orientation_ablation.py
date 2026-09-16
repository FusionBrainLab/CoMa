"""Run the no-orientation contextual-relevance ablation.

The script retrains the learned CR metric without orientation-related features,
validates it on the metric-selection benchmark, optionally rescors any generated
method outputs materialized under ``methods_validation/``, then renders the
publication visualizations from ``results_no_orientation/``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COMA = ROOT / "experiments" / "benchmarks" / "coma"
HYDRA_RUN = ROOT / "hydra_run.py"
TRAIN_CONFIG = COMA / "metrics_validation" / "train_ensemble_no_orientation_config.yaml"
EVALUATE_SCRIPT = COMA / "metrics_validation" / "evaluate_learned_metric_artifact.py"
METHOD_GRID_CONFIG = COMA / "score_methods_no_orientation_config.yaml"
VIS_SCRIPT = COMA / "results_visualization" / "scripts" / "publication" / "no_orientation_visualizations.sh"


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT, env={**os.environ, "HYDRA_FULL_ERROR": "1", "REPO_ROOT": str(ROOT)})


def hydra(config: Path, *overrides: str) -> None:
    run([sys.executable, str(HYDRA_RUN), "--config_path", str(config), *overrides])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull-dvc", action="store_true", help="Fetch DVC-managed metric artifacts first.")
    parser.add_argument("--skip-methods", action="store_true", help="Do not rescore generated method outputs.")
    parser.add_argument("--skip-visualizations", action="store_true", help="Do not render publication figures.")
    args = parser.parse_args()

    if args.pull_dvc:
        run(["dvc", "pull", str(COMA / "metrics_validation" / "results.dvc"), str(COMA / "metrics_validation" / "models.dvc")])

    hydra(TRAIN_CONFIG)
    run([sys.executable, str(EVALUATE_SCRIPT)])

    if not args.skip_methods:
        hydra(METHOD_GRID_CONFIG)

    if not args.skip_visualizations:
        run(["bash", str(VIS_SCRIPT)])


if __name__ == "__main__":
    main()
