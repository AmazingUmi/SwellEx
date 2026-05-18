"""HDF5 loading for standalone SCM-GRNN reference and prediction data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch


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


def infer_layout(h5: h5py.File, x_shape: tuple[int, ...], n_labels: int, path: Path) -> H5Layout:
    ri_candidates = [axis for axis, size in enumerate(x_shape) if size == 2]
    sample_candidates = [axis for axis, size in enumerate(x_shape) if size == n_labels]

    if len(x_shape) != 4:
        raise ValueError(f"{path} /X must be 4-D, got shape {x_shape}.")
    if not ri_candidates:
        raise ValueError(f"{path} /X has no real/imag axis of length 2: {x_shape}.")
    if not sample_candidates:
        raise ValueError(
            f"{path} /X has no sample axis matching /y_range_km length {n_labels}: "
            f"{x_shape}."
        )

    ri_axis = ri_candidates[-1]
    sample_axis = sample_candidates[0]
    if ri_axis == sample_axis and len(sample_candidates) > 1:
        sample_axis = sample_candidates[1]
    if ri_axis == sample_axis:
        raise ValueError(f"{path} cannot distinguish sample and real/imag axes: {x_shape}.")

    remaining_axes = [axis for axis in range(4) if axis not in (sample_axis, ri_axis)]
    if len(remaining_axes) != 2:
        raise ValueError(f"{path} cannot infer pair/frequency axes: {x_shape}.")

    element_count = int(max(h5["/array/depth_m"].shape)) if "/array/depth_m" in h5 else 0
    freq_count = (
        int(max(h5["/frequency/freq_hz"].shape))
        if "/frequency/freq_hz" in h5
        else None
    )
    pair_count = (
        int(max(h5["/pair/numerator_element_idx"].shape))
        if "/pair/numerator_element_idx" in h5
        else None
    )

    freq_axis = None
    pair_axis = None
    if freq_count is not None:
        for axis in remaining_axes:
            if x_shape[axis] == freq_count:
                freq_axis = axis
                break
    if pair_count is not None:
        for axis in remaining_axes:
            if x_shape[axis] == pair_count:
                pair_axis = axis
                break

    if pair_axis is None and freq_axis is not None:
        pair_axis = next(axis for axis in remaining_axes if axis != freq_axis)
    if freq_axis is None and pair_axis is not None:
        freq_axis = next(axis for axis in remaining_axes if axis != pair_axis)
    if pair_axis is None or freq_axis is None:
        freq_axis = max(remaining_axes, key=lambda axis: x_shape[axis])
        pair_axis = next(axis for axis in remaining_axes if axis != freq_axis)

    n_pairs = int(x_shape[pair_axis])
    if pair_count is not None and pair_count != n_pairs:
        raise ValueError(f"{path} pair metadata has {pair_count} pairs but /X has {n_pairs}.")

    return H5Layout(
        sample_axis=sample_axis,
        pair_axis=pair_axis,
        freq_axis=freq_axis,
        ri_axis=ri_axis,
        n_samples=int(x_shape[sample_axis]),
        n_elements=element_count,
        n_pairs=n_pairs,
        n_freq_bins=int(x_shape[freq_axis]),
    )


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
    paths = [Path(path) for path in h5_paths]
    features: list[np.ndarray] = []
    labels: list[float] = []
    source_segment_idx: list[int] = []
    window_center_s: list[float] = []
    input_shape: tuple[int, int, int] | None = None

    for path in paths:
        with h5py.File(path, "r") as h5:
            if "/X" not in h5 or "/y_range_km" not in h5:
                raise KeyError(f"{path} must contain /X and /y_range_km.")

            y = np.asarray(h5["/y_range_km"]).reshape(-1).astype(np.float32)
            x_shape = tuple(int(size) for size in h5["/X"].shape)
            layout = infer_layout(h5, x_shape, len(y), path)
            shape_chw = (2, layout.n_pairs, layout.n_freq_bins)
            if input_shape is None:
                input_shape = shape_chw
            elif input_shape != shape_chw:
                raise ValueError(
                    f"All files must share input shape. Expected {input_shape}, "
                    f"got {shape_chw} in {path}."
                )

            valid = np.ones_like(y, dtype=bool)
            if "/valid_sample" in h5:
                valid &= np.asarray(h5["/valid_sample"]).reshape(-1).astype(bool)
            valid &= np.isfinite(y)

            if "/split/source_segment_idx" in h5:
                source_idx = np.asarray(h5["/split/source_segment_idx"]).reshape(-1)
            else:
                source_idx = np.arange(1, len(y) + 1, dtype=np.uint64)
            if "/time/window_center_s" in h5:
                window_s = np.asarray(h5["/time/window_center_s"]).reshape(-1)
            else:
                window_s = source_idx.astype(np.float64)

            for sample_idx in np.flatnonzero(valid):
                selection: list[int | slice] = [slice(None)] * 4
                selection[layout.sample_axis] = int(sample_idx)
                raw = np.asarray(h5["/X"][tuple(selection)], dtype=np.float32)
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

    if input_shape is None or not features:
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
    )
