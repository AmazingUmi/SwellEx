"""Method-aware HDF5 loading for GRNN reference and prediction data."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import h5py

from scripts.localization_networks.grnn.bindings import validate_method


@dataclass
class GrnnFeatureBundle:
    x: torch.Tensor
    y_range_km: torch.Tensor
    source_segment_idx: list[int]
    window_center_s: list[float]
    input_shape: tuple[int, int, int]
    source_paths: list[Path]
    input_signature: dict[str, object]


def channel_normalize(x_chw: np.ndarray) -> np.ndarray:
    """Preserve GRNN's existing per-channel population-standard-deviation rule."""
    mean = x_chw.mean(axis=(1, 2), keepdims=True)
    std = x_chw.std(axis=(1, 2), keepdims=True)
    return (x_chw - mean) / np.maximum(std, 1.0e-6)


def load_grnn_features(
    h5_paths: Iterable[Path],
    *,
    method: str,
    per_sample_channel_norm: bool = False,
    max_samples: int | None = None,
) -> GrnnFeatureBundle:
    """Flatten canonical ``[real_imag, element|pair, frequency]`` samples.

    Method loaders own native/reversed layouts, physical-axis signatures, pair
    semantics, valid-label filtering, and identities. Their neural normalization
    is disabled here because GRNN retains its population-standard-deviation
    preprocessing and defaults to the original real/imaginary features.
    """
    method = validate_method(method)
    if type(per_sample_channel_norm) is not bool:
        raise ValueError("per_sample_channel_norm must be a boolean.")
    if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
        raise ValueError("max_samples must be a positive integer.")
    paths = [Path(path) for path in h5_paths]
    if method == "RBD":
        for path in paths:
            with h5py.File(path, "r") as handle:
                if "/pair" in handle:
                    raise ValueError(f"{path} RBD-GRNN input cannot contain pair metadata.")
    module = import_module(f"scripts.localization_networks.{method.lower()}.network.data")
    dataset_cls = getattr(module, f"{method.title()}RangeH5Dataset")
    dataset = dataset_cls(paths, normalize_input=False)
    try:
        if dataset.input_shape is None or dataset.input_signature is None:
            raise ValueError(f"No valid {method} samples found in the provided HDF5 files.")
        count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        features: list[np.ndarray] = []
        for index in range(count):
            x, _ = dataset[index]
            x_chw = x.numpy()
            if per_sample_channel_norm:
                # Reconstruct the previous storage-order transpose view before
                # NumPy reductions, preserving its float32 summation order.
                file_index, _ = dataset.samples[index]
                layout = dataset.layouts[file_index]
                feature_axis = layout.element_axis if method == "RBD" else layout.pair_axis
                canonical_axes = (layout.ri_axis, feature_axis, layout.freq_axis)
                stored_axes = [axis for axis in range(4) if axis != layout.sample_axis]
                storage_order = tuple(canonical_axes.index(axis) for axis in stored_axes)
                inverse_order = tuple(storage_order.index(axis) for axis in range(3))
                x_chw = x_chw.transpose(storage_order).copy().transpose(inverse_order)
                x_chw = channel_normalize(x_chw)
            features.append(x_chw.reshape(-1).astype(np.float32, copy=False))
        return GrnnFeatureBundle(
            x=torch.from_numpy(np.stack(features, axis=0)).to(dtype=torch.float32),
            y_range_km=torch.tensor(dataset.labels[:count], dtype=torch.float32).view(-1, 1),
            source_segment_idx=dataset.source_segment_idx[:count],
            window_center_s=dataset.window_center_s[:count],
            input_shape=dataset.input_shape,
            source_paths=paths,
            input_signature=dataset.input_signature,
        )
    finally:
        dataset.close()
