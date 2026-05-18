import argparse
import json
import warnings

from src.config_utils import load_config

warnings.filterwarnings("ignore")

def main(config):
    method = load_config(config)
    method()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
    main(config)