"""Shared HDF5 dataset helpers."""

from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Iterable, Protocol

import torch


class HasLabels(Protocol):
    labels: list[float]


def resolve_h5_paths(patterns: Iterable[str], required: bool = True) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            path = Path(pattern)
            if path.is_file():
                paths.append(path)
    paths = sorted({p.resolve() for p in paths})
    if required and not paths:
        raise FileNotFoundError(
            "No HDF5 files matched. Pass one or more files or glob patterns with --data."
        )
    return paths


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
    n_val = min(n_val, n - 1) if n > 1 else 0
    return indices[n_val:], indices[:n_val]


def subset_labels(dataset: HasLabels, indices: list[int]) -> torch.Tensor:
    values = [dataset.labels[i] for i in indices]
    return torch.tensor(values, dtype=torch.float32).view(-1, 1)
