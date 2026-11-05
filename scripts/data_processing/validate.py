#!/usr/bin/env python3
"""Validate generator numerical contracts, or generated HDF5 datasets via `datasets --root ... --report ...`."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data_processing.numerical_validation import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli())
