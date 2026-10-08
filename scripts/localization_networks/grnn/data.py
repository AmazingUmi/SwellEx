"""HDF5 loading for standalone SCM-GRNN reference and prediction data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch

from scripts.localization_networks.common.h5_utils import metadata_layout_axes
from scripts.localization_networks.common.input_contract import (
    read_input_signature, require_matching_input_signature,
)


@dataclass(frozen=True)
class H5Layout:
    sample_axis: int
    pair_axis: int
    freq_axis: int
    ri_axis: int
    n_samples: int
    n_elements: int
    n_pairs: int
    n_freq_bins: int


@dataclass
class GrnnFeatureBundle:
    x: torch.Tensor
    y_range_km: torch.Tensor
    source_segment_idx: list[int]
    window_center_s: list[float]
    input_shape: tuple[int, int, int]
    source_paths: list[Path]
    input_signature: dict[str, object]


def infer_layout(h5: h5py.File, x_shape: tuple[int, ...], n_labels: int, path: Path) -> H5Layout:
    declared = metadata_layout_axes(h5, x_shape, n_labels, "pair", path)
    if declared is not None:
        sample, pair, frequency, ri = declared
        elements = int(h5["/array/depth_m"].size) if "/array/depth_m" in h5 else 0
        return H5Layout(sample, pair, frequency, ri, x_shape[sample],
                        elements, x_shape[pair], x_shape[frequency])
    if len(x_shape) != 4:
        raise ValueError(f"{path} /X must be 4-D, got shape {x_shape}.")
    required = ("/array/depth_m", "/frequency/freq_hz", "/pair/numerator_element_idx")
    if any(key not in h5 for key in required):
        raise ValueError(f"{path} lacks metadata required to resolve SCM axes.")
    elements = int(h5["/array/depth_m"].size)
    pairs = int(h5["/pair/numerator_element_idx"].size)
    frequencies = int(h5["/frequency/freq_hz"].size)
    logical_shape = (n_labels, pairs, frequencies, 2)
    candidates = []
    if x_shape == logical_shape:
        candidates.append((0, 1, 2, 3))
    if x_shape == logical_shape[::-1]:
        candidates.append((3, 2, 1, 0))
    if len(candidates) != 1:
        raise ValueError(f"{path} cannot uniquely resolve native/reversed SCM axes: {x_shape}; declare storage_axis_order.")
    sample, pair, frequency, ri = candidates[0]
    return H5Layout(sample, pair, frequency, ri, n_labels, elements, pairs, frequencies)


def sample_to_chw(raw: np.ndarray, layout: H5Layout) -> np.ndarray:
    remaining_axes = [axis for axis in range(4) if axis != layout.sample_axis]
    ri_pos = remaining_axes.index(layout.ri_axis)
    pair_pos = remaining_axes.index(layout.pair_axis)
    freq_pos = remaining_axes.index(layout.freq_axis)
    return np.transpose(raw, (ri_pos, pair_pos, freq_pos))


def channel_normalize(x_chw: np.ndarray) -> np.ndarray:
    mean = x_chw.mean(axis=(1, 2), keepdims=True)
    std = x_chw.std(axis=(1, 2), keepdims=True)
    return (x_chw - mean) / np.maximum(std, 1.0e-6)


def load_scm_grnn_features(
    h5_paths: Iterable[Path],
    *,
    per_sample_channel_norm: bool = False,
    max_samples: int | None = None,
) -> GrnnFeatureBundle:
    if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
        raise ValueError("max_samples must be a positive integer.")
    paths = [Path(path) for path in h5_paths]
    features: list[np.ndarray] = []
    labels: list[float] = []
    source_segment_idx: list[int] = []
    window_center_s: list[float] = []
    input_shape: tuple[int, int, int] | None = None
    input_signature: dict[str, object] | None = None

    for path in paths:
        with h5py.File(path, "r") as h5:
            if "/X" not in h5 or "/y_range_km" not in h5:
                raise KeyError(f"{path} must contain /X and /y_range_km.")

            y = np.asarray(h5["/y_range_km"]).reshape(-1).astype(np.float32)
            x_shape = tuple(int(size) for size in h5["/X"].shape)
            layout = infer_layout(h5, x_shape, len(y), path)
            shape_chw = (2, layout.n_pairs, layout.n_freq_bins)
            signature = read_input_signature(h5, shape_chw, "SCM")
            if input_signature is None:
                input_signature = signature
            else:
                require_matching_input_signature(input_signature, signature, str(path))
            if input_shape is None:
                input_shape = shape_chw
            elif input_shape != shape_chw:
                raise ValueError(
                    f"All files must share input shape. Expected {input_shape}, "
                    f"got {shape_chw} in {path}."
                )

            valid = np.ones_like(y, dtype=bool)
            if "/valid_sample" in h5:
                mask = np.asarray(h5["/valid_sample"]).reshape(-1)
                if len(mask) != len(y) or not np.all(np.isin(mask, [0, 1])):
                    raise ValueError(f"{path} /valid_sample must contain one 0/1 value per label.")
                valid &= mask.astype(bool)
            valid &= np.isfinite(y)

            if "/split/source_segment_idx" in h5:
                source_idx = np.asarray(h5["/split/source_segment_idx"]).reshape(-1)
            else:
                source_idx = np.arange(1, len(y) + 1, dtype=np.uint64)
            if "/time/window_center_s" in h5:
                window_s = np.asarray(h5["/time/window_center_s"]).reshape(-1)
            else:
                window_s = source_idx.astype(np.float64)

            if len(source_idx) != len(y) or len(window_s) != len(y):
                raise ValueError(f"{path} sample identity/time vector lengths must match labels.")
            if not np.all(np.isfinite(source_idx)) or not np.all(source_idx >= 1) or not np.all(source_idx == np.floor(source_idx)):
                raise ValueError(f"{path} source segment indices must be positive integers.")
            if not np.all(np.isfinite(window_s)):
                raise ValueError(f"{path} window center times must be finite.")
            for sample_idx in np.flatnonzero(valid):
                selection: list[int | slice] = [slice(None)] * 4
                selection[layout.sample_axis] = int(sample_idx)
                raw = np.asarray(h5["/X"][tuple(selection)], dtype=np.float32)
                if not np.all(np.isfinite(raw)):
                    raise ValueError(f"{path} nonfinite SCM feature at sample {sample_idx + 1}.")
                x_chw = sample_to_chw(raw, layout)
                if per_sample_channel_norm:
                    x_chw = channel_normalize(x_chw)
                features.append(x_chw.reshape(-1).astype(np.float32, copy=False))
                labels.append(float(y[sample_idx]))
                source_segment_idx.append(int(source_idx[sample_idx]))
                window_center_s.append(float(window_s[sample_idx]))
                if max_samples is not None and len(features) >= max_samples:
                    break

        if max_samples is not None and len(features) >= max_samples:
            break

    if input_shape is None or input_signature is None or not features:
        raise ValueError("No valid SCM samples found in the provided HDF5 files.")

    x = torch.from_numpy(np.stack(features, axis=0)).to(dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return GrnnFeatureBundle(
        x=x,
        y_range_km=y_tensor,
        source_segment_idx=source_segment_idx,
        window_center_s=window_center_s,
        input_shape=input_shape,
        source_paths=paths,
        input_signature=input_signature,
    )
