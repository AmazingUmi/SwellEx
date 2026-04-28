"""Command-line interface for reusable range-regression workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .paths import (
    DEFAULT_OUTPUT_DIR,
)


MODEL_CHOICES = ["complex_cnn_range", "real_cnn_range"]


try:
    import torch
except ModuleNotFoundError:
    torch = None


def dependency_error(exc: ModuleNotFoundError) -> SystemExit:
    return SystemExit(
        f"Missing Python dependency: {exc.name}. Install dependencies with:\n"
        "  python -m pip install -r scripts_py/requirements.txt\n"
        "or install PyTorch with the command matching your CUDA/CPU setup."
    )


def default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_command(args: argparse.Namespace) -> None:
    try:
        from .training import train
    except ModuleNotFoundError as exc:
        raise dependency_error(exc) from exc
    train(args)


def predict_command(args: argparse.Namespace) -> None:
    try:
        from .prediction import predict
    except ModuleNotFoundError as exc:
        raise dependency_error(exc) from exc
    predict(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Complex CNN range regression for RBD HDF5 datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a range regressor.")
    train_parser.add_argument(
        "--data",
        required=True,
        help="Dataset code under outputs/Datasets, for example periodic_4_1.",
    )
    train_parser.add_argument(
        "--train-data",
        nargs="+",
        default=None,
        help="Optional override for training HDF5 files or glob patterns.",
    )
    train_parser.add_argument(
        "--val-data",
        nargs="+",
        default=None,
        help=(
            "Optional override for validation HDF5 files or glob patterns. "
            "If omitted, *_val.h5 is used when present."
        ),
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Base output directory. Results are saved under "
            "<output-dir>/<model>/train_outputs unless --output-dir already names the model."
        ),
    )
    train_parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="Model architecture to train.",
    )
    train_parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint to continue training from. If omitted, the newest "
            "last_*.pt under <output-dir>/<model>/train_outputs is used when available."
        ),
    )
    train_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a fresh training run even if a previous checkpoint exists.",
    )
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=3.0e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    train_parser.add_argument("--base-channels", type=int, default=16)
    train_parser.add_argument("--dropout", type=float, default=0.15)
    train_parser.add_argument("--huber-beta", type=float, default=0.5)
    train_parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.25,
        help="Validation fraction split from training data. Default: 0.25 for 3:1 train/val.",
    )
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument(
        "--device",
        default=default_device(),
        help="cpu, cuda, or cuda:0.",
    )
    train_parser.add_argument(
        "--no-input-norm",
        action="store_true",
        help="Disable per-sample normalization of real and imaginary channels.",
    )
    train_parser.set_defaults(func=train_command)

    predict_parser = subparsers.add_parser(
        "predict", help="Run a saved checkpoint on HDF5 files."
    )
    predict_parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="Model architecture whose default checkpoint/output directory should be used.",
    )
    predict_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Base output directory. Default checkpoint and prediction files are "
            "resolved under <output-dir>/<model>."
        ),
    )
    predict_parser.add_argument(
        "--data",
        required=True,
        help="Dataset code under outputs/Datasets, for example periodic_4_1.",
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    predict_parser.add_argument("--batch-size", type=int, default=32)
    predict_parser.add_argument("--num-workers", type=int, default=0)
    predict_parser.add_argument(
        "--device",
        default=default_device(),
        help="cpu, cuda, or cuda:0.",
    )
    predict_parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
    )
    predict_parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Path for the label-line/prediction-scatter plot.",
    )
    predict_parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable prediction plot generation.",
    )
    predict_parser.add_argument("--print-first", type=int, default=10)
    predict_parser.set_defaults(func=predict_command)

    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        raise SystemExit(2)
    else:
        args = parser.parse_args()
    args.func(args)
