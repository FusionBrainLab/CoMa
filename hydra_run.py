import sys
import argparse
import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

sys.path.append("/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation")


def main(cfg: DictConfig):
    method = instantiate(cfg.method)
    method()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", type=str)
    parser.add_argument("--config_path", dest="config_path_arg", type=str)
    args, hydra_args = parser.parse_known_args()

    config_path = args.config_path_arg or args.config_path
    if config_path is None:
        parser.error("config_path is required")

    sys.argv = [sys.argv[0], *hydra_args]
    
    config_file = Path(config_path).expanduser().resolve()
    config_path = os.path.relpath(config_file.parent, Path(__file__).resolve().parent)
    config_name = config_file.stem

    hydra_main = hydra.main(
        version_base=None,
        config_path=config_path,
        config_name=config_name,
    )(main)

    hydra_main()