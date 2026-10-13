"""Lazy native joint HDF5 datasets and leakage-free block scaling."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from scripts.joint_feature_pack.localization_networks.common.h5_utils import split_indices, subset_labels
from scripts.joint_feature_pack.localization_networks.common.input_contract import require_matching_input_signature
from .contract import validate_preprocessing, validate_signature, validate_resume_data_identity


class JointRangeH5Dataset(Dataset):
    def __init__(self, h5_paths: Iterable[Path], normalize_input: bool = True, *,
                 method: str | None = None, preprocessing_state: dict | None = None) -> None:
        self._handles = {}
        self.h5_paths = [Path(path) for path in h5_paths]
        self.normalize_input = normalize_input
        self.samples = []
        self.labels = []
        self.source_segment_idx = []
        self.window_center_s = []
        self.input_shape = None
        self.input_signature = None
        self.preprocessing_state = None
        for file_id, path in enumerate(self.h5_paths):
            with h5py.File(path, "r") as handle:
                if handle.attrs.get("pack") != "joint_feature_pack":
                    raise ValueError("Joint loaders require joint_feature_pack HDF5 data.")
                raw_shape = tuple(handle["X"].shape)
                order = "sample,feature,frequency,real_imag"
                if (len(raw_shape) != 4 or raw_shape[-1] != 2
                        or handle["X"].attrs.get("axis_order") != order
                        or handle["X"].attrs.get("storage_axis_order") != order):
                    raise ValueError("Joint HDF5 requires native sample,feature,frequency,real_imag layout.")
                shape = (2, raw_shape[1], raw_shape[2])
                signature = json.loads(handle.attrs["fusion_input_signature"])
                validate_signature(signature, shape, method)
                if self.input_signature is None:
                    self.input_signature, self.input_shape = signature, shape
                else:
                    require_matching_input_signature(self.input_signature, signature, str(path))
                values = {key: np.asarray(handle[key]).reshape(-1) for key in
                          ("y_range_km", "valid_sample", "split/source_segment_idx", "time/window_center_s",
                           "time/window_start_s", "time/window_stop_s", "time/sample_start_idx", "time/sample_stop_idx")}
                if any(len(v) != raw_shape[0] or not np.isfinite(v).all() for v in values.values()):
                    raise ValueError("Joint sample labels/identities/windows must be finite and match X.")
                source = values["split/source_segment_idx"]
                if (not np.all(values["valid_sample"] == 1) or (source < 1).any()
                        or not np.equal(source, np.floor(source)).all()
                        or len(np.unique(source)) != len(source)):
                    raise ValueError("Joint rows must be valid and have unique positive integer source IDs.")
                if ((values["time/window_stop_s"] <= values["time/window_start_s"]).any()
                        or (values["time/window_center_s"] < values["time/window_start_s"]).any()
                        or (values["time/window_center_s"] > values["time/window_stop_s"]).any()):
                    raise ValueError("Joint sample windows are inconsistent.")
                self.samples.extend((file_id, index) for index in range(raw_shape[0]))
                self.labels.extend(values["y_range_km"].astype(float).tolist())
                self.source_segment_idx.extend(source.astype(int).tolist())
                self.window_center_s.extend(values["time/window_center_s"].astype(float).tolist())
        if not self.samples:
            raise ValueError("No valid joint samples found.")
        if len(set(self.source_segment_idx)) != len(self.samples):
            raise ValueError("Joint source identities are duplicated across input files.")
        if preprocessing_state is not None:
            self.set_preprocessing(preprocessing_state)

    def __len__(self):
        return len(self.samples)

    def raw_array(self, index: int) -> np.ndarray:
        file_id, sample_id = self.samples[index]
        if file_id not in self._handles:
            self._handles[file_id] = h5py.File(self.h5_paths[file_id], "r")
        values = np.asarray(self._handles[file_id]["X"][sample_id], dtype=np.float32).transpose(2, 0, 1)
        if not np.isfinite(values).all():
            raise ValueError("Joint features contain nonfinite values.")
        return values

    def set_preprocessing(self, state: dict) -> None:
        validate_preprocessing(state, self.input_signature)
        self.preprocessing_state = deepcopy(state)

    def __getitem__(self, index):
        values = self.raw_array(index).copy()
        if self.preprocessing_state is not None:
            for block, scale in zip(self.input_signature["blocks"], self.preprocessing_state["scales"]):
                values[:, block["start"]:block["stop"], :] /= scale
        x = torch.from_numpy(values).contiguous()
        if self.normalize_input:
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True, correction=int(x[0].numel() > 1)).clamp_min(1.0e-6)
            x = (x - mean) / std
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Joint preprocessing produced nonfinite features.")
        return x, torch.tensor([self.labels[index]], dtype=torch.float32)

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):
        self.close()


def fit_preprocessing(dataset: JointRangeH5Dataset, indices: list[int]) -> dict:
    if not indices:
        raise ValueError("Block RMS requires nonempty actual training indices.")
    signature = dataset.input_signature
    sums = [0.0] * len(signature["blocks"])
    if signature["fusion_strategy"] == "block_rms":
        for index in indices:
            values = dataset.raw_array(index).astype(np.float64)
            for block_id, block in enumerate(signature["blocks"]):
                sums[block_id] += float(np.square(values[:, block["start"]:block["stop"], :]).sum())
        scales = [max(math.sqrt(total / (len(indices) * (block["stop"] - block["start"]) * dataset.input_shape[2])), 1e-12)
                  for total, block in zip(sums, signature["blocks"])]
    else:
        scales = [1.0] * len(sums)
    state = {"strategy": signature["fusion_strategy"], "scales": scales, "fit_sample_count": len(indices)}
    validate_preprocessing(state, signature)
    return state


def fit_flat_preprocessing(features: torch.Tensor, signature: dict) -> dict:
    """Fit raw flattened CHW blocks using only the caller's reference rows."""
    shape = signature["input_shape"]
    values = features.reshape(-1, *shape).to(dtype=torch.float64)
    if values.size(0) == 0 or not torch.isfinite(values).all():
        raise ValueError("Block RMS references must be nonempty and finite.")
    scales = []
    for block in signature["blocks"]:
        if signature["fusion_strategy"] == "raw_concat":
            scales.append(1.0)
        else:
            raw = values[:, :, block["start"]:block["stop"], :]
            # Real and imaginary parts share one complex RMS per physical block.
            scales.append(max(float(raw.square().sum(dim=1).mean().sqrt()), 1e-12))
    state = {"strategy": signature["fusion_strategy"], "scales": scales, "fit_sample_count": values.size(0)}
    validate_preprocessing(state, signature)
    return state


def apply_flat_preprocessing(features: torch.Tensor, signature: dict, state: dict) -> torch.Tensor:
    validate_preprocessing(state, signature)
    values = features.reshape(-1, *signature["input_shape"]).clone()
    for block, scale in zip(signature["blocks"], state["scales"]):
        values[:, :, block["start"]:block["stop"], :] /= scale
    if not torch.isfinite(values).all():
        raise ValueError("Block scaling produced nonfinite features.")
    return values.reshape(features.shape)


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
    input_preprocessing: dict
    resume_data_identity: dict


def training_source_identity(train_paths: list[Path], val_paths: list[Path]) -> dict:
    """Fingerprint every byte of each ordered train/val source HDF5 file.

    The roles share a cached digest for a random split of the same files.
    Absolute paths and complete content hashes prevent same-path rewrites or
    row reorderings from silently changing a resumed split and target stats.
    """
    cache = {}

    def sources(paths):
        result = []
        for source in paths:
            path = Path(source).resolve(strict=True)
            if path not in cache:
                before = path.stat()
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    raise ValueError(f"Training source changed while hashing: {path}")
                cache[path] = {"path": str(path), "size_bytes": after.st_size, "sha256": digest.hexdigest()}
            result.append(dict(cache[path]))
        return result

    identity = {"format_version": 1, "train_sources": sources(train_paths), "val_sources": sources(val_paths)}
    validate_resume_data_identity(identity)
    return identity


def random_split_bundle(source_paths, *, method, normalize_input, val_fraction, seed, split_mode):
    dataset = JointRangeH5Dataset(source_paths, normalize_input=normalize_input, method=method)
    train_idx, val_idx = split_indices(len(dataset), val_fraction, seed)
    state = fit_preprocessing(dataset, train_idx)
    dataset.close()  # Worker processes must open their own HDF5 handles after RMS fitting.
    dataset.set_preprocessing(state)
    identity = training_source_identity(source_paths, source_paths)
    return DatasetBundle(dataset, Subset(dataset, train_idx), Subset(dataset, val_idx), train_idx, val_idx,
                         subset_labels(dataset, train_idx), dataset.input_shape, split_mode, source_paths,
                         dataset.input_signature, state, identity)


def fixed_split_bundle(train_paths, val_paths, *, method, normalize_input, split_mode):
    train = JointRangeH5Dataset(train_paths, normalize_input=normalize_input, method=method)
    val = JointRangeH5Dataset(val_paths, normalize_input=normalize_input, method=method)
    require_matching_input_signature(train.input_signature, val.input_signature, "Joint train/validation")
    if set(train.source_segment_idx) & set(val.source_segment_idx):
        raise ValueError("Joint training and validation source identities overlap.")
    train_idx, val_idx = list(range(len(train))), list(range(len(val)))
    state = fit_preprocessing(train, train_idx)
    train.close()
    train.set_preprocessing(state)
    val.set_preprocessing(state)
    identity = training_source_identity(train_paths, val_paths)
    return DatasetBundle(train, train, val, train_idx, val_idx, subset_labels(train, train_idx),
                         train.input_shape, split_mode, [*train_paths, *val_paths], train.input_signature, state, identity)


def data_bindings(method: str):
    return SimpleNamespace(random_split_bundle=partial(random_split_bundle, method=method),
                           fixed_split_bundle=partial(fixed_split_bundle, method=method))
