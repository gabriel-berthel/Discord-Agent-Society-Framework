import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import yaml


def load_agent_logs(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
        return SimpleNamespace(**obj)


def save_benchmark_results(results):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            return super().default(obj)

    with open("outputs/qa_bench/results.json", "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)


def load_yaml(file_path):
    """Loads a YAML file and returns its contents as a dictionary."""

    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error loading YAML file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)
