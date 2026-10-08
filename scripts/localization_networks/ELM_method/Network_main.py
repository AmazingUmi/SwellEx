"""Command-line entry point for ELM pairwise-ratio range regression.

This script exposes the training and prediction pipeline implemented in
`scripts/localization_networks/ELM_method/network/`.

Expected HDF5 layout:
    Files are produced by `scripts/data_processing/generate.py`.

    outputs/Datasets/<split_strategy>/*_train.h5
    outputs/Datasets/<split_strategy>/*_test.h5
    /X                         [window, pair, frequency, real_imag]
    /pair/numerator_element_idx [pair, 1]
    /pair/denominator_element_idx [pair, 1]
    /y_range_km                [window, 1]
    /valid_sample              [window, 1], optional
    /split/source_segment_idx  [window, 1], optional
    /time/window_center_s      [window, 1], optional

The ELM network input is:
    torch input = [batch, 2, pair, frequency]
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.localization_networks.ELM_method.network.cli import main
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency: {exc.name}. Install dependencies with:\n"
        "  uv sync --frozen\n"
        "Run network commands through uv run from the project root."
    ) from exc


if __name__ == "__main__":
    main()
