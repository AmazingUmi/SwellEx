"""Command-line entry point for ELM pairwise-ratio range regression.

This script exposes the training and prediction pipeline implemented in
`scripts_py/ELM_method/network/`.

Expected HDF5 layout:
    Files are produced by `scripts_matlab/Signals_Segmentation_elm.m`.

    outputs/Datasets/<split_strategy>/*_train.h5
    outputs/Datasets/<split_strategy>/*_test.h5
    /X                         [window, numerator_element,
                                denominator_element, frequency, real_imag]
    /y_range_km                [window, 1]
    /valid_sample              [window, 1], optional
    /split/source_segment_idx  [window, 1], optional
    /time/window_center_s      [window, 1], optional

The first ELM network path flattens the pair grid:
    torch input = [batch, 2, element_pair, frequency]
    where element_pair = numerator_element * denominator_element.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_PY_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_PY_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PY_DIR))

try:
    from network.cli import main
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency: {exc.name}. Install dependencies with:\n"
        "  python -m pip install -r scripts_py/requirements.txt\n"
        "or install PyTorch with the command matching your CUDA/CPU setup."
    ) from exc


if __name__ == "__main__":
    main()
