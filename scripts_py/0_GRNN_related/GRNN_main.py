"""Standalone entry point for SCM-GRNN reference and prediction workflows."""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SCRIPTS_PY_DIR = THIS_DIR.parent
for path in (THIS_DIR, SCRIPTS_PY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from grnn_cli import main
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency: {exc.name}. Install dependencies with:\n"
        "  python -m pip install -r scripts_py/requirements.txt\n"
        "or install PyTorch with the command matching your CUDA/CPU setup."
    ) from exc


if __name__ == "__main__":
    main()
