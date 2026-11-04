"""Canonical generation configuration for base_pack."""
from __future__ import annotations

from scripts.data_processing.common.generation_config import (
    DATASET_DEFAULTS,
    FREQUENCY_KEYS,
    MULTIPATH_DEFAULTS,
    RBD_DEFAULTS,
    ROOT,
    normalize_config,
    output_path,
    read_config,
)

__all__ = [
    "DATASET_DEFAULTS", "FREQUENCY_KEYS", "MULTIPATH_DEFAULTS", "RBD_DEFAULTS",
    "ROOT", "normalize_config", "output_path", "read_config",
]
