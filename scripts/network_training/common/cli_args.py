"""Shared argparse value types for neural and GRNN command lines."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

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
