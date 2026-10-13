"""Command-line interface for reusable range-regression workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import math

from .bindings import MethodBindings


import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pdf_path(value: str) -> Path:
    path = Path(value)
    if path.suffix.lower() != ".pdf":
        raise argparse.ArgumentTypeError("prediction plot path must end in .pdf")
    return path


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def seed_int(value: str) -> int:
    number = int(value)
    if not 0 <= number <= 2**32 - 1:
        raise argparse.ArgumentTypeError("seed must be between 0 and 4294967295")
    return number


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def positive_float(value: str) -> float:
    number = _finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_float(value: str) -> float:
    number = _finite_float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def fraction_float(value: str) -> float:
    number = _finite_float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("must be strictly between 0 and 1")
    return number


def dropout_float(value: str) -> float:
    number = _finite_float(value)
    if not 0 <= number < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return number


def run_train(args: argparse.Namespace, bindings: MethodBindings) -> None:
    from .training import train
    train(args, bindings)


def run_predict(args: argparse.Namespace, bindings: MethodBindings) -> None:
    from .prediction import predict
    predict(args, bindings)


def build_parser(bindings: MethodBindings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Range regression for {bindings.method} HDF5 datasets.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", allow_abbrev=False, help="Train a range regressor.")
    train_parser.add_argument(
        "--data",
        required=True,
        help="Dataset code under outputs/Datasets, for example full_interleaved_4_1.",
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
        default=bindings.paths.DEFAULT_OUTPUT_DIR,
        help=(
            "Base output directory. Results are saved under "
            "<output-dir>/<model>/train_outputs unless --output-dir already names the model."
        ),
    )
    train_parser.add_argument(
        "--model",
        required=True,
        choices=sorted(bindings.model.MODEL_REGISTRY),
        help="Model architecture to train.",
    )
    resume_group = train_parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint to continue training from. If omitted, the newest "
            "last_*.pt under <output-dir>/<model>/train_outputs is used when available."
        ),
    )
    resume_group.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a fresh training run even if a previous checkpoint exists.",
    )
    train_parser.add_argument("--epochs", type=positive_int, default=100)
    train_parser.add_argument("--batch-size", type=positive_int, default=16)
    train_parser.add_argument("--lr", type=positive_float, default=3.0e-4)
    train_parser.add_argument("--weight-decay", type=nonnegative_float, default=1.0e-4)
    train_parser.add_argument("--base-channels", type=positive_int, default=16)
    train_parser.add_argument("--dropout", type=dropout_float, default=0.15)
    train_parser.add_argument("--huber-beta", type=nonnegative_float, default=0.5)
    train_parser.add_argument(
        "--loss-space",
        choices=["normalized", "km"],
        default="normalized",
        help=(
            "Space used to compute the training loss. 'normalized' matches the "
            "current target-normalized behavior; 'km' computes loss directly on "
            "physical range error in kilometers."
        ),
    )
    train_parser.add_argument(
        "--val-fraction",
        type=fraction_float,
        default=0.25,
        help="Validation fraction split from training data. Default: 0.25 for 3:1 train/val.",
    )
    train_parser.add_argument("--seed", type=seed_int, default=2026)
    train_parser.add_argument("--num-workers", type=nonnegative_int, default=0)
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
    train_parser.set_defaults(func=lambda args: run_train(args, bindings))

    predict_parser = subparsers.add_parser(
        "predict", allow_abbrev=False, help="Run a saved checkpoint on HDF5 files."
    )
    predict_parser.add_argument(
        "--model",
        required=True,
        choices=sorted(bindings.model.MODEL_REGISTRY),
        help="Model architecture whose default checkpoint/output directory should be used.",
    )
    predict_parser.add_argument(
        "--output-dir",
        type=Path,
        default=bindings.paths.DEFAULT_OUTPUT_DIR,
        help=(
            "Base output directory. Default checkpoint and prediction files are "
            "resolved under <output-dir>/<model>."
        ),
    )
    predict_parser.add_argument(
        "--data",
        required=True,
        help="Dataset code under outputs/Datasets, for example full_interleaved_4_1.",
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    predict_parser.add_argument("--batch-size", type=positive_int, default=32)
    predict_parser.add_argument("--num-workers", type=nonnegative_int, default=0)
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
        type=pdf_path,
        default=None,
        help="PDF path for label and prediction markers; all samples are retained.",
    )
    predict_parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable prediction plot generation.",
    )
    predict_parser.add_argument("--print-first", type=nonnegative_int, default=10)
    predict_parser.set_defaults(func=lambda args: run_predict(args, bindings))

    return parser


def main(bindings: MethodBindings, argv: list[str] | None = None) -> None:
    parser = build_parser(bindings)
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        parser.print_help()
        raise SystemExit(2)
    else:
        args = parser.parse_args(argv)
    args.func(args)
