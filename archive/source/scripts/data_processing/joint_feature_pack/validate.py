#!/usr/bin/env python3
"""Validate joint_feature_pack numerical contracts or generated HDF5 datasets."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.data_processing.joint_feature_pack.generation_config import normalize_config
from scripts.data_processing.joint_feature_pack.generation_core import GENERATOR_VERSION, PACK_DIR
from scripts.data_processing.common.numerical_validation import run_cli, run_validation as _run_validation


def run_validation(report_path):
    return _run_validation(report_path, normalize_config_fn=normalize_config,
                           generator_version=GENERATOR_VERSION, extra_manifest_dirs=(PACK_DIR,))


def main(argv: list[str] | None = None) -> int:
    return run_cli(argv, normalize_config_fn=normalize_config,
                   generator_version=GENERATOR_VERSION, pack_dir=PACK_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
