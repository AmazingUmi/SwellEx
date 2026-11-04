"""Canonical generation configuration for MoE_data_scale."""
from __future__ import annotations

from scripts.data_processing.common.generation_config import (
    DATASET_DEFAULTS,
    FREQUENCY_KEYS,
    MULTIPATH_DEFAULTS,
    RBD_DEFAULTS,
    ROOT,
    normalize_config as _normalize_config,
    output_path,
    read_config as _read_config,
)

PACK = "MoE_data_scale"


def normalize_config(raw, root=ROOT):
    return _normalize_config(raw, root, forbid_adapt_pack=PACK)


def read_config(path, root=ROOT):
    return _read_config(path, root, forbid_adapt_pack=PACK)


__all__ = [
    "DATASET_DEFAULTS", "FREQUENCY_KEYS", "MULTIPATH_DEFAULTS", "RBD_DEFAULTS",
    "ROOT", "normalize_config", "output_path", "read_config",
]
