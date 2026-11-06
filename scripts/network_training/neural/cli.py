"""Command-line interface for the shared RBD/SCM neural range-regression workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.data_processing.input_contract import METHODS
from scripts.network_training.common.cli_args import (
    default_device, dropout_float, fraction_float, nonnegative_float, nonnegative_int,
    pdf_path, positive_float, positive_int, seed_int,
)
from scripts.network_training.common.paths import default_output_dir

from .model import MODEL_METHOD, MODEL_REGISTRY


class _NeuralParser(argparse.ArgumentParser):
    """Reject models of another method and resolve the per-method default output root."""

    def parse_args(self, args=None, namespace=None):
        result = super().parse_args(args, namespace)
        if MODEL_METHOD[result.model] != result.method:
            self.error(f"model {result.model} belongs to {MODEL_METHOD[result.model]}, not {result.method}")
        if result.output_dir is None:
            result.output_dir = default_output_dir(result.method)
        return result


def run_train(args: argparse.Namespace) -> None:
    from .bindings import bindings_for
    from .training import train
    train(args, bindings_for(args.method))


def run_predict(args: argparse.Namespace) -> None:
    from .bindings import bindings_for
    from .prediction import predict
    predict(args, bindings_for(args.method))


def build_parser() -> argparse.ArgumentParser:
    parser = _NeuralParser(
        description="Range regression for RBD/SCM HDF5 datasets.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", allow_abbrev=False, help="Train a range regressor.")
    train_parser.add_argument("--method", required=True, choices=METHODS, help="Feature route.")
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
        default=None,
        help=(
            "Base output directory (default outputs/networks_results/<METHOD>). Results are saved under "
            "<output-dir>/<model>/train_outputs unless --output-dir already names the model."
        ),
    )
    train_parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_REGISTRY),
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
    train_parser.add_argument(
        "--val-split-mode",
        choices=["random", "blocked"],
        default="random",
        help="How to split validation samples from training HDF5 when --val-data is not given.",
    )
    train_parser.add_argument(
        "--val-gap-s",
        type=nonnegative_float,
        default=0.0,
        help="Minimum window-center gap (seconds) between train and validation in blocked split mode.",
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
    train_parser.set_defaults(func=run_train)

    predict_parser = subparsers.add_parser(
        "predict", allow_abbrev=False, help="Run a saved checkpoint on HDF5 files."
    )
    predict_parser.add_argument("--method", required=True, choices=METHODS, help="Feature route.")
    predict_parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_REGISTRY),
        help="Model architecture whose default checkpoint/output directory should be used.",
    )
    predict_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Base output directory (default outputs/networks_results/<METHOD>). Default checkpoint "
            "and prediction files are resolved under <output-dir>/<model>."
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
    predict_parser.set_defaults(func=run_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        parser.print_help()
        raise SystemExit(2)
    args = parser.parse_args(argv)
    args.func(args)
