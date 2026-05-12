"""HDF5 dataset utilities for MATLAB-exported ELM pairwise-ratio features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from common.h5_utils import resolve_h5_paths, split_indices, subset_labels


@dataclass(frozen=True)
class H5Layout:
    sample_axis: int
    numerator_axis: int
    denominator_axis: int
    freq_axis: int
    ri_axis: int
    n_samples: int
    n_elements: int
    n_pairs: int
    n_freq_bins: int


@dataclass(frozen=True)
class DatasetBundle:
    dataset: Dataset
    train_dataset: Dataset
    val_dataset: Dataset
    train_idx: list[int]
    val_idx: list[int]
    train_labels: torch.Tensor
    input_shape: tuple[int, int, int]
    pair_grid_shape: tuple[int, int]
    split_mode: str
    source_paths: list[Path]


def random_split_bundle(
    source_paths: list[Path],
    *,
    normalize_input: bool,
    val_fraction: float,
    seed: int,
    split_mode: str,
) -> DatasetBundle:
    dataset = ElmRangeH5Dataset(source_paths, normalize_input=normalize_input)
    if dataset.input_shape is None:
        raise RuntimeError("Dataset input shape was not initialized.")

    train_idx, val_idx = split_indices(len(dataset), val_fraction, seed)
    train_labels = subset_labels(dataset, train_idx)
    return DatasetBundle(
        dataset=dataset,
        train_dataset=Subset(dataset, train_idx),
        val_dataset=Subset(dataset, val_idx),
        train_idx=train_idx,
        val_idx=val_idx,
        train_labels=train_labels,
        input_shape=dataset.input_shape,
        pair_grid_shape=dataset.pair_grid_shape,
        split_mode=split_mode,
        source_paths=source_paths,
    )


def fixed_split_bundle(
    train_paths: list[Path],
    val_paths: list[Path],
    *,
    normalize_input: bool,
    split_mode: str,
) -> DatasetBundle:
    train_dataset = ElmRangeH5Dataset(train_paths, normalize_input=normalize_input)
    val_dataset = ElmRangeH5Dataset(val_paths, normalize_input=normalize_input)
    if train_dataset.input_shape is None or val_dataset.input_shape is None:
        raise RuntimeError("Dataset input shape was not initialized.")
    if train_dataset.input_shape != val_dataset.input_shape:
        raise ValueError(
            "Train and validation files must share input shape. "
            f"Train has {train_dataset.input_shape}, val has {val_dataset.input_shape}."
        )
    if train_dataset.pair_grid_shape != val_dataset.pair_grid_shape:
        raise ValueError(
            "Train and validation files must share pair grid shape. "
            f"Train has {train_dataset.pair_grid_shape}, "
            f"val has {val_dataset.pair_grid_shape}."
        )

    train_idx = list(range(len(train_dataset)))
    val_idx = list(range(len(val_dataset)))
    return DatasetBundle(
        dataset=train_dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        train_labels=torch.tensor(train_dataset.labels, dtype=torch.float32).view(-1, 1),
        input_shape=train_dataset.input_shape,
        pair_grid_shape=train_dataset.pair_grid_shape,
        split_mode=split_mode,
        source_paths=[*train_paths, *val_paths],
    )


class ElmRangeH5Dataset(Dataset):
    """Lazy HDF5 dataset for ELM pairwise element-ratio features."""

    def __init__(self, h5_paths: Iterable[Path], normalize_input: bool = True) -> None:
        self.h5_paths = [Path(p) for p in h5_paths]
        self.normalize_input = normalize_input
        self.samples: list[tuple[int, int]] = []
        self.labels: list[float] = []
        self.source_segment_idx: list[int] = []
        self.window_center_s: list[float] = []
        self.layouts: list[H5Layout] = []
        self.input_shape: tuple[int, int, int] | None = None
        self.pair_grid_shape: tuple[int, int] = (0, 0)
        self._handles: dict[int, h5py.File] = {}

        for file_idx, path in enumerate(self.h5_paths):
            with h5py.File(path, "r") as h5:
                if "/X" not in h5 or "/y_range_km" not in h5:
                    raise KeyError(f"{path} must contain /X and /y_range_km.")

                x_shape = tuple(int(v) for v in h5["/X"].shape)
                if len(x_shape) != 5 or 2 not in x_shape:
                    raise ValueError(
                        f"{path} /X must be 5-D with one real/imag axis of length 2, "
                        f"got {x_shape}."
                    )

                y = np.asarray(h5["/y_range_km"]).reshape(-1).astype(np.float32)
                layout = self._infer_layout(h5, x_shape, len(y), path)
                self.layouts.append(layout)

                shape_chw = (2, layout.n_pairs, layout.n_freq_bins)
                pair_grid_shape = (layout.n_elements, layout.n_elements)
                if self.input_shape is None:
                    self.input_shape = shape_chw
                    self.pair_grid_shape = pair_grid_shape
                elif self.input_shape != shape_chw:
                    raise ValueError(
                        f"All files must share input shape. Expected {self.input_shape}, "
                        f"got {shape_chw} in {path}."
                    )
                elif self.pair_grid_shape != pair_grid_shape:
                    raise ValueError(
                        "All files must share pair grid shape. "
                        f"Expected {self.pair_grid_shape}, got {pair_grid_shape} in {path}."
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
                    window_center_s = np.asarray(h5["/time/window_center_s"]).reshape(-1)
                else:
                    window_center_s = source_idx.astype(np.float64)

                for sample_idx in np.flatnonzero(valid):
                    self.samples.append((file_idx, int(sample_idx)))
                    self.labels.append(float(y[sample_idx]))
                    self.source_segment_idx.append(int(source_idx[sample_idx]))
                    self.window_center_s.append(float(window_center_s[sample_idx]))

        if not self.samples:
            raise ValueError("No valid samples found in the provided HDF5 files.")

    @staticmethod
    def _infer_layout(
        h5: h5py.File, x_shape: tuple[int, int, int, int, int], n_labels: int, path: Path
    ) -> H5Layout:
        ri_candidates = [axis for axis, size in enumerate(x_shape) if size == 2]
        sample_candidates = [axis for axis, size in enumerate(x_shape) if size == n_labels]

        if not ri_candidates:
            raise ValueError(f"{path} /X has no real/imag axis of length 2: {x_shape}.")
        if not sample_candidates:
            raise ValueError(
                f"{path} /X has no sample axis matching /y_range_km length "
                f"{n_labels}: {x_shape}."
            )

        ri_axis = ri_candidates[-1]
        sample_axis = sample_candidates[0]
        if ri_axis == sample_axis and len(sample_candidates) > 1:
            sample_axis = sample_candidates[1]
        if ri_axis == sample_axis:
            raise ValueError(f"{path} cannot distinguish sample and real/imag axes: {x_shape}.")

        remaining_axes = [
            axis for axis in range(5) if axis not in (ri_axis, sample_axis)
        ]
        if len(remaining_axes) != 3:
            raise ValueError(
                f"{path} cannot infer numerator/denominator/frequency axes: {x_shape}."
            )

        element_count = (
            int(max(h5["/array/depth_m"].shape)) if "/array/depth_m" in h5 else None
        )
        freq_count = (
            int(max(h5["/frequency/freq_hz"].shape))
            if "/frequency/freq_hz" in h5
            else None
        )

        if freq_count is not None:
            freq_matches = [axis for axis in remaining_axes if x_shape[axis] == freq_count]
        else:
            freq_matches = []
        if freq_matches:
            freq_axis = freq_matches[0]
        else:
            freq_axis = max(remaining_axes, key=lambda axis: x_shape[axis])

        pair_axes = [axis for axis in remaining_axes if axis != freq_axis]
        if len(pair_axes) != 2:
            raise ValueError(f"{path} cannot infer pair axes: {x_shape}.")

        if element_count is not None:
            bad_axes = [axis for axis in pair_axes if x_shape[axis] != element_count]
            if bad_axes:
                raise ValueError(
                    f"{path} pair axes must match array element count {element_count}, "
                    f"got {x_shape}."
                )

        numerator_axis, denominator_axis = pair_axes
        n_elements = x_shape[numerator_axis]
        if x_shape[denominator_axis] != n_elements:
            raise ValueError(
                f"{path} numerator and denominator axes must have equal size, got {x_shape}."
            )

        return H5Layout(
            sample_axis=sample_axis,
            numerator_axis=numerator_axis,
            denominator_axis=denominator_axis,
            freq_axis=freq_axis,
            ri_axis=ri_axis,
            n_samples=x_shape[sample_axis],
            n_elements=n_elements,
            n_pairs=n_elements * n_elements,
            n_freq_bins=x_shape[freq_axis],
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _h5(self, file_idx: int) -> h5py.File:
        handle = self._handles.get(file_idx)
        if handle is None:
            handle = h5py.File(self.h5_paths[file_idx], "r")
            self._handles[file_idx] = handle
        return handle

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx, sample_idx = self.samples[idx]
        h5 = self._h5(file_idx)
        layout = self.layouts[file_idx]

        selection: list[int | slice] = [slice(None)] * 5
        selection[layout.sample_axis] = sample_idx
        raw = np.asarray(h5["/X"][tuple(selection)], dtype=np.float32)

        remaining_axes = [axis for axis in range(5) if axis != layout.sample_axis]
        numerator_pos = remaining_axes.index(layout.numerator_axis)
        denominator_pos = remaining_axes.index(layout.denominator_axis)
        freq_pos = remaining_axes.index(layout.freq_axis)
        ri_pos = remaining_axes.index(layout.ri_axis)

        # Reorder to [numerator, denominator, frequency, real_imag], then flatten
        # the pair grid to [pair, frequency, real_imag].
        x_np = np.transpose(raw, (numerator_pos, denominator_pos, freq_pos, ri_pos))
        x_np = x_np.reshape(layout.n_pairs, layout.n_freq_bins, 2)
        x = torch.from_numpy(x_np).permute(2, 0, 1).contiguous()

        if self.normalize_input:
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
            x = (x - mean) / std

        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()
