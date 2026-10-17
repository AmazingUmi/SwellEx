"""Lazy native joint HDF5 datasets and leakage-free block scaling."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from scripts.MoE_data_scale.localization_networks.common.h5_utils import split_indices, subset_labels
from scripts.MoE_data_scale.localization_networks.common.input_contract import require_matching_input_signature
from ..contract import validate_preprocessing, validate_signature


class JointRangeH5Dataset(Dataset):
    def __init__(self, h5_paths: Iterable[Path], normalize_input: bool = True, *,
                 method: str = "RBD_SCM", preprocessing_state: dict | None = None) -> None:
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
                if handle.attrs.get("pack") != "MoE_data_scale":
                    raise ValueError("Joint loaders require MoE_data_scale HDF5 data.")
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
                for key in ("time/sample_start_idx", "time/sample_stop_idx"):
                    if (values[key] < 1).any() or not np.equal(values[key], np.floor(values[key])).all():
                        raise ValueError("Joint sample indices must be positive 1-based integers.")
                if (values["time/sample_stop_idx"] < values["time/sample_start_idx"]).any():
                    raise ValueError("Joint sample stop must be inclusive and follow sample start.")
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
        if self.preprocessing_state is None:
            raise ValueError("RBD_SCM samples require fitted or restored block_rms preprocessing.")
        values = self.raw_array(index).copy()
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
    if (not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(dataset) for i in indices)):
        raise ValueError("Block RMS requires nonempty actual training indices.")
    signature = dataset.input_signature
    sums = [0.0] * len(signature["blocks"])
    for index in indices:
        values = dataset.raw_array(index).astype(np.float64)
        for block_id, block in enumerate(signature["blocks"]):
            sums[block_id] += float(np.square(values[:, block["start"]:block["stop"], :]).sum())
    scales = [max(math.sqrt(total / (len(indices) * (block["stop"] - block["start"]) * dataset.input_shape[2])), 1e-12)
              for total, block in zip(sums, signature["blocks"])]
    state = {"strategy": signature["fusion_strategy"], "scales": scales, "fit_sample_count": len(indices),
             "fit_source_identity_sha256": hashlib.sha256(
                 ",".join(str(dataset.source_segment_idx[i]) for i in indices).encode()).hexdigest()}
    validate_preprocessing(state, signature)
    return state


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


def random_split_bundle(source_paths, *, normalize_input, val_fraction, seed, split_mode):
    dataset = JointRangeH5Dataset(source_paths, normalize_input=normalize_input)
    train_idx, val_idx = split_indices(len(dataset), val_fraction, seed)
    return DatasetBundle(dataset, Subset(dataset, train_idx), Subset(dataset, val_idx), train_idx, val_idx,
                         subset_labels(dataset, train_idx), dataset.input_shape, split_mode, source_paths,
                         dataset.input_signature)


def fixed_split_bundle(train_paths, val_paths, *, normalize_input, split_mode):
    train = JointRangeH5Dataset(train_paths, normalize_input=normalize_input)
    val = JointRangeH5Dataset(val_paths, normalize_input=normalize_input)
    require_matching_input_signature(train.input_signature, val.input_signature, "Joint train/validation")
    if set(train.source_segment_idx) & set(val.source_segment_idx):
        raise ValueError("Joint training and validation source identities overlap.")
    train_idx, val_idx = list(range(len(train))), list(range(len(val)))
    return DatasetBundle(train, train, val, train_idx, val_idx, subset_labels(train, train_idx),
                         train.input_shape, split_mode, [*train_paths, *val_paths], train.input_signature)


def prepare_preprocessing(bundle, sampling: dict, saved_state: dict | None = None) -> dict:
    """Fit exclusively on retained optimizer rows; continuation restores saved scales."""
    state = fit_preprocessing(bundle.dataset, bundle.train_idx) if saved_state is None else deepcopy(saved_state)
    validate_preprocessing(state, bundle.input_signature)
    if (state["fit_sample_count"] != len(bundle.train_idx)
            or state["fit_source_identity_sha256"] != sampling["source_identity_sha256"]):
        raise ValueError("Joint preprocessing fitted membership differs from retained optimizer training rows.")
    datasets = [bundle.dataset]
    validation = bundle.val_dataset.dataset if isinstance(bundle.val_dataset, Subset) else bundle.val_dataset
    if validation is not bundle.dataset:
        datasets.append(validation)
    for dataset in datasets:
        dataset.close()
        dataset.set_preprocessing(state)
    return state
