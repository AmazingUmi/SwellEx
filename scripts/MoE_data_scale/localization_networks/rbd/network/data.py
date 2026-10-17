"""HDF5 dataset utilities for native or MATLAB-reversed RBD features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from scripts.MoE_data_scale.localization_networks.common.input_contract import (
    read_input_signature, require_matching_input_signature,
)

from scripts.MoE_data_scale.localization_networks.common.h5_utils import (
    metadata_layout_axes, resolve_h5_paths, split_indices, subset_labels,
)


@dataclass(frozen=True)
class H5Layout:
    sample_axis: int
    element_axis: int
    freq_axis: int
    ri_axis: int
    n_samples: int
    n_elements: int
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
    split_mode: str
    source_paths: list[Path]
    input_signature: dict


def random_split_bundle(
    source_paths: list[Path],
    *,
    normalize_input: bool,
    val_fraction: float,
    seed: int,
    split_mode: str,
) -> DatasetBundle:
    dataset = RbdRangeH5Dataset(source_paths, normalize_input=normalize_input)
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
        split_mode=split_mode,
        source_paths=source_paths,
        input_signature=dataset.input_signature,
    )


def fixed_split_bundle(
    train_paths: list[Path],
    val_paths: list[Path],
    *,
    normalize_input: bool,
    split_mode: str,
) -> DatasetBundle:
    train_dataset = RbdRangeH5Dataset(train_paths, normalize_input=normalize_input)
    val_dataset = RbdRangeH5Dataset(val_paths, normalize_input=normalize_input)
    if train_dataset.input_shape is None or val_dataset.input_shape is None:
        raise RuntimeError("Dataset input shape was not initialized.")
    if train_dataset.input_shape != val_dataset.input_shape:
        raise ValueError(
            "Train and validation files must share input shape. "
            f"Train has {train_dataset.input_shape}, val has {val_dataset.input_shape}."
        )

    require_matching_input_signature(train_dataset.input_signature, val_dataset.input_signature,
                                     "Train/validation files")
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
        split_mode=split_mode,
        source_paths=[*train_paths, *val_paths],
        input_signature=train_dataset.input_signature,
    )


class RbdRangeH5Dataset(Dataset):
    """Lazy HDF5 dataset for native or MATLAB-reversed RBD Green's function features."""

    def __init__(self, h5_paths: Iterable[Path], normalize_input: bool = True) -> None:
        self.h5_paths = [Path(p) for p in h5_paths]
        self.normalize_input = normalize_input
        self.samples: list[tuple[int, int]] = []
        self.labels: list[float] = []
        self.source_segment_idx: list[int] = []
        self.window_center_s: list[float] = []
        self.layouts: list[H5Layout] = []
        self.input_shape: tuple[int, int, int] | None = None
        self.input_signature: dict | None = None
        self._handles: dict[int, h5py.File] = {}

        for file_idx, path in enumerate(self.h5_paths):
            with h5py.File(path, "r") as h5:
                if "/X" not in h5 or "/y_range_km" not in h5:
                    raise KeyError(f"{path} must contain /X and /y_range_km.")

                x_shape = tuple(int(v) for v in h5["/X"].shape)
                if len(x_shape) != 4 or x_shape[-1] != 2:
                    if len(x_shape) != 4 or 2 not in x_shape:
                        raise ValueError(
                            f"{path} /X must contain one real/imag axis of length 2, "
                            f"got {x_shape}."
                        )

                y = np.asarray(h5["/y_range_km"]).reshape(-1).astype(np.float32)
                layout = self._infer_layout(h5, x_shape, len(y), path)
                self.layouts.append(layout)

                shape_chw = (2, layout.n_elements, layout.n_freq_bins)
                if self.input_shape is None:
                    self.input_shape = shape_chw
                elif self.input_shape != shape_chw:
                    raise ValueError(
                        f"All files must share input shape. Expected {self.input_shape}, "
                        f"got {shape_chw} in {path}."
                    )

                signature = read_input_signature(h5, shape_chw, "RBD")
                if self.input_signature is None:
                    self.input_signature = signature
                else:
                    require_matching_input_signature(self.input_signature, signature, str(path))

                valid = np.ones_like(y, dtype=bool)
                if "/valid_sample" in h5:
                    mask = np.asarray(h5["/valid_sample"]).reshape(-1)
                    if len(mask) != len(y) or not np.isin(mask, [0, 1]).all():
                        raise ValueError(f"{path} /valid_sample must be a binary vector matching labels.")
                    valid &= mask.astype(bool)
                valid &= np.isfinite(y)
                if "/split/source_segment_idx" in h5:
                    source_idx = np.asarray(h5["/split/source_segment_idx"]).reshape(-1)
                else:
                    raise ValueError(f"{path} requires /split/source_segment_idx for training-count identity")
                if "/time/window_center_s" in h5:
                    window_center_s = np.asarray(h5["/time/window_center_s"]).reshape(-1)
                else:
                    raise ValueError(f"{path} requires /time/window_center_s for temporal sampling")

                if len(source_idx) != len(y) or not np.isfinite(source_idx).all() or (source_idx < 1).any() or not np.equal(source_idx, np.floor(source_idx)).all():
                    raise ValueError(f"{path} source segment IDs must be positive integers matching labels.")
                if len(window_center_s) != len(y) or not np.isfinite(window_center_s).all():
                    raise ValueError(f"{path} window times must be finite and match labels.")

                for sample_idx in np.flatnonzero(valid):
                    self.samples.append((file_idx, int(sample_idx)))
                    self.labels.append(float(y[sample_idx]))
                    self.source_segment_idx.append(int(source_idx[sample_idx]))
                    self.window_center_s.append(float(window_center_s[sample_idx]))

        if not self.samples:
            raise ValueError("No valid samples found in the provided HDF5 files.")

    @staticmethod
    def _infer_layout(
        h5: h5py.File, x_shape: tuple[int, int, int, int], n_labels: int, path: Path
    ) -> H5Layout:
        declared = metadata_layout_axes(h5, x_shape, n_labels, "element", path)
        if declared is not None:
            sample, element, frequency, ri = declared
            return H5Layout(sample, element, frequency, ri, x_shape[sample],
                            x_shape[element], x_shape[frequency])
        raise ValueError(f"{path} input layout metadata is incomplete.")

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

        selection: list[int | slice] = [slice(None)] * 4
        selection[layout.sample_axis] = sample_idx
        raw = np.asarray(h5["/X"][tuple(selection)], dtype=np.float32)

        remaining_axes = [axis for axis in range(4) if axis != layout.sample_axis]
        element_pos = remaining_axes.index(layout.element_axis)
        freq_pos = remaining_axes.index(layout.freq_axis)
        ri_pos = remaining_axes.index(layout.ri_axis)

        # Reorder to [element, frequency, real_imag], then PyTorch [channel, H, W].
        x_np = np.transpose(raw, (element_pos, freq_pos, ri_pos))
        if not np.isfinite(x_np).all():
            raise ValueError(f"{self.h5_paths[file_idx]} sample {sample_idx} contains nonfinite features.")
        x = torch.from_numpy(x_np).permute(2, 0, 1).contiguous()

        if self.normalize_input:
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True, correction=int(x[0].numel() > 1)).clamp_min(1.0e-6)
            x = (x - mean) / std
            if not bool(torch.isfinite(x).all()):
                raise ValueError(f"{self.h5_paths[file_idx]} sample {sample_idx} normalization is nonfinite.")

        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()
