"""Shared HDF5 dataset helpers."""

from __future__ import annotations

import glob
import math
import random
from pathlib import Path
from typing import Iterable, Protocol

import torch


def metadata_layout_axes(h5, x_shape: tuple[int, ...], n_labels: int,
                         feature_name: str, path: Path) -> tuple[int, int, int, int] | None:
    """Resolve declared HDF5 axes before using legacy dimension heuristics.

    axis_order describes logical axes; storage_axis_order describes actual
    h5py order when supplied. Without declarations, complete metadata must
    unambiguously match native or reversed storage; dimension guesses are rejected.
    """
    attrs = h5["/X"].attrs
    if len(x_shape) != 4:
        raise ValueError(f"{path} /X must be four-dimensional, got {x_shape}.")

    def text(value) -> str:
        if hasattr(value, "size") and value.size == 1:
            value = value.item()
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    logical = ("sample", feature_name, "frequency", "real_imag")
    if "axis_order" in attrs and text(attrs["axis_order"]) != ",".join(logical):
        raise ValueError(f"{path} /X axis_order must be {','.join(logical)}.")
    feature_key = "/array/depth_m" if feature_name == "element" else "/pair/numerator_element_idx"
    if feature_key not in h5 or "/frequency/freq_hz" not in h5:
        return None
    feature_count = int(h5[feature_key].size)
    frequency_count = int(h5["/frequency/freq_hz"].size)
    logical_shape = (n_labels, feature_count, frequency_count, 2)
    if "storage_axis_order" in attrs:
        stored = tuple(text(attrs["storage_axis_order"]).split(","))
        if len(stored) != 4 or set(stored) != set(logical):
            raise ValueError(f"{path} /X storage_axis_order must declare each logical axis once.")
        axes = tuple(stored.index(axis) for axis in logical)
        if tuple(x_shape[axis] for axis in axes) != logical_shape:
            raise ValueError(f"{path} /X shape {x_shape} disagrees with declared axes and metadata {logical_shape}.")
        return axes
    if tuple(x_shape) == logical_shape == logical_shape[::-1]:
        raise ValueError(f"{path} ambiguous native/reversed axes require storage_axis_order.")
    if tuple(x_shape) == logical_shape:
        return (0, 1, 2, 3)
    if tuple(x_shape) == logical_shape[::-1]:
        return (3, 2, 1, 0)
    raise ValueError(f"{path} /X shape {x_shape} differs from logical {logical_shape} and reversed {logical_shape[::-1]}.")


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
    if not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be finite and strictly between 0 and 1.")
    if n < 2:
        raise ValueError("Random training/validation split requires at least two valid samples.")
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = min(max(1, int(round(n * val_fraction))), n - 1)
    return indices[n_val:], indices[:n_val]


def subset_labels(dataset: HasLabels, indices: list[int]) -> torch.Tensor:
    values = [dataset.labels[i] for i in indices]
    return torch.tensor(values, dtype=torch.float32).view(-1, 1)
