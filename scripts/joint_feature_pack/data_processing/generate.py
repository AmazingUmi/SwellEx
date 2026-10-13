#!/usr/bin/env python3
"""Generate RBD/ELM/SCM datasets with NumPy, SciPy and h5py (run via uv)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.joint_feature_pack.data_processing.generation_config import ROOT, read_config
from scripts.joint_feature_pack.data_processing.generation_core import generate_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = read_config(args.config)
        report = generate_datasets(config)
    except Exception as failure:
        print(f"Dataset generation failed: {type(failure).__name__}: {failure}", file=sys.stderr)
        return 1
    print(f"Generation {report['status']}: {ROOT / config['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
