"""Reproducibility, seeding, and config utilities."""

import hashlib
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def set_global_seed(seed: int = 42) -> None:
    """Fix random seeds across Python and NumPy for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str | Path = "configs/experiment_config.yaml") -> Dict[str, Any]:
    """Load YAML experiment configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_md5(file_path: str | Path) -> str:
    """Compute MD5 checksum of a file for data/artifact verification in reports."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

