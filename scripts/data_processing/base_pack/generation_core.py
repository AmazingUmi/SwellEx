"""Feature generation orchestration for base_pack."""
from __future__ import annotations

from pathlib import Path

from scripts.data_processing.common.generation_config import ROOT, normalize_config, output_path
from scripts.data_processing.common.generation_core import (
    GENERATOR_VERSION,
    create_h5,
    dataset_names,
    generate_datasets as _generate_datasets,
    generate_method,
    generation_identity,
    json_value,
    load_channel,
    load_inputs,
    source_manifest as _source_manifest,
    write_json,
)

PACK_DIR = Path(__file__).resolve().parent


def source_manifest():
    return _source_manifest((PACK_DIR,))


def generate_datasets(raw_config, root=ROOT):
    return _generate_datasets(raw_config, root, normalize_config_fn=normalize_config, extra_manifest_dirs=(PACK_DIR,))


__all__ = [
    "GENERATOR_VERSION", "ROOT", "create_h5", "dataset_names", "generate_datasets",
    "generate_method", "generation_identity", "json_value", "load_channel",
    "load_inputs", "output_path", "source_manifest", "write_json",
]
