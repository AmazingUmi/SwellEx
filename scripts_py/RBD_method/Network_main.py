"""Command-line entry point for reusable range-regression network workflows.

This script exposes the shared training and prediction pipeline implemented in
`scripts_py/RBD_method/network/`. It is intentionally thin: command-line parsing,
dataset loading, model construction, checkpoint resume, prediction export, and
plot generation all live in reusable modules so different model architectures
and data split strategies can share the same workflow.

Subcommands:
    train
        Train a registered model on explicitly provided HDF5 files. Pass
        `--model` and either `--train-data` or `--data`. The train command
        creates a seeded random train/validation split unless `--val-data` is
        provided, and automatically resumes from the newest
        `outputs/networks_results/RBD_method/<model_name>/train_outputs/<dataset>_last_*.pt` checkpoint when
        one exists. Use `--no-resume` to force a fresh run, `--val-data` for an
        explicit validation set, and `--resume-checkpoint` for a specific file.

    predict
        Run inference for an explicitly specified model and dataset. Pass
        `--model` and `--data`. If `--checkpoint` is omitted, the command uses
        `outputs/networks_results/RBD_method/<model_name>/train_outputs/<dataset>_best.pt`; if that
        fixed path is missing, the newest matching timestamped best checkpoint
        is used.

Expected HDF5 layout:
    Files are produced by `scripts_matlab/Signals_Segmentation.m`.

    outputs/Datasets/<split_strategy>/*_train.h5
    outputs/Datasets/<split_strategy>/*_test.h5
    /X                         [window, element, frequency, real_imag]
    /y_range_km                [window, 1]
    /valid_sample              [window, 1], optional
    /split/source_segment_idx  [window, 1], optional
    /time/window_center_s      [window, 1], optional

Output naming:
    Training checkpoints and histories are written with a compact time suffix:
        outputs/networks_results/RBD_method/<model_name>/train_outputs/<dataset>_best_MMDD_HHMMSS.pt
        outputs/networks_results/RBD_method/<model_name>/train_outputs/<dataset>_last_MMDD_HHMMSS.pt
        outputs/networks_results/RBD_method/<model_name>/train_outputs/<dataset>_history_MMDD_HHMMSS.json

    Prediction files use the same suffix rule:
        outputs/networks_results/RBD_method/<model_name>/test_outputs/<dataset>_predictions_MMDD_HHMMSS.csv
        outputs/networks_results/RBD_method/<model_name>/test_outputs/<dataset>_range_prediction_MMDD_HHMMSS.png

Extension point:
    Add new models in separate `network/models/model_*.py` files, then register
    their config class and module class in `network/model.py`. The shared
    training, resume, and prediction code should continue to import only from
    `network.model`.
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
