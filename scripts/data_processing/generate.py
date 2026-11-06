#!/usr/bin/env python3
"""Generate RBD/SCM datasets with NumPy, SciPy and h5py (run via uv)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data_processing.generation_config import ROOT, read_config
from scripts.data_processing.generation_core import generate_datasets
from scripts.data_processing.input_contract import METHODS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--method",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Feature route(s) to generate on demand",
    )
    args = parser.parse_args(argv)
    try:
        if len(set(args.method)) != len(args.method):
            raise ValueError("Duplicate --method entries are not allowed")
        config = read_config(args.config, methods=args.method)
        report = generate_datasets(config, methods=args.method)
    except Exception as failure:
        print(f"Dataset generation failed: {type(failure).__name__}: {failure}", file=sys.stderr)
        return 1
    print(f"Generation {report['status']}: {ROOT / config['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
