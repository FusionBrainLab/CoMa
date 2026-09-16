import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
HYDRA_RUN_PATH = str(SCRIPT_DIR / "hydra_run.py")
sys.path.insert(0, str(SCRIPT_DIR))
BASE_CONFIG = ""


def format_override_value(value: Any) -> str:
    value = OmegaConf.to_container(value, resolve=True) if isinstance(value, (DictConfig, ListConfig)) else value
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ",".join(format_override_value(item) for item in value) + "]"
    return str(value)


def flatten_overrides(cfg: DictConfig, prefix: str | None = None) -> list[str]:
    overrides = []
    for key, value in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, DictConfig):
            overrides.extend(flatten_overrides(value, full_key))
        else:
            overrides.append(f"{full_key}={format_override_value(value)}")
    return overrides


def main(cfg: DictConfig) -> None:
    processes = []
    for _, run_cfg in cfg.runs.items():
        if run_cfg is None:
            continue
        command = [
            sys.executable,
            HYDRA_RUN_PATH,
            "--config_path",
            BASE_CONFIG,
            *flatten_overrides(run_cfg),
        ]
        processes.append(subprocess.Popen(command))

    return_codes = [process.wait() for process in processes]
    failed = [code for code in return_codes if code != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} batch runs failed.")


if __name__ == "__main__":
    os.environ.setdefault("REPO_ROOT", str(SCRIPT_DIR))

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True, type=str)
    parser.add_argument("--batch_config", required=True, type=str)
    args, hydra_args = parser.parse_known_args()

    BASE_CONFIG = str(Path(args.base_config).expanduser().resolve())
    sys.argv = [sys.argv[0], *hydra_args]

    config_file = Path(args.batch_config).expanduser().resolve()
    config_path = os.path.relpath(config_file.parent, SCRIPT_DIR)
    config_name = config_file.stem

    hydra_main = hydra.main(
        version_base=None,
        config_path=config_path,
        config_name=config_name,
    )(main)

    hydra_main()
