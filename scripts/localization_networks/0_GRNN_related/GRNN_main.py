"""Standalone entry point for SCM-GRNN reference and prediction workflows."""

from __future__ import annotations

import sys
import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    main = importlib.import_module("scripts.localization_networks.0_GRNN_related.grnn_cli").main
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency: {exc.name}. Install dependencies with:\n"
        "  uv sync --frozen\n"
        "Run network commands through uv run from the project root."
    ) from exc


if __name__ == "__main__":
    main()
