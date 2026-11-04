"""Window-centered RBD/SCM feature generation for continual MoE_learn stages."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.data_processing.common.feature_math import extend_profile
from scripts.data_processing.common.generation_core import (
    _input_file_identity,
    create_h5 as _create_h5_common,
    dataset_names as _dataset_names_common,
    generate_datasets as _generate_datasets,
    generate_method as _generate_method_common,
    generation_identity as _generation_identity_common,
    json_value,
    load_channel,
    source_manifest as _source_manifest,
    write_json,
)
from scripts.data_processing.MoE_learn.generation_config import ROOT, normalize_config, output_path

GENERATOR_VERSION = "moe-learn-python-features-v3-window-centered"
PACK_DIR = Path(__file__).resolve().parent


def source_manifest():
    return _source_manifest((PACK_DIR,))


def load_inputs(root, config):
    """Load raw recordings without full-record centering so windows remain causal."""
    origin = Path(root)/"Origindata"
    d = config["dataset"]; event = d["event_name"]
    positions_path = origin/"positions/positions_vla.txt"
    positions = np.loadtxt(positions_path, ndmin=2)
    if positions.shape[1] < 2 or positions.shape[0] < 2 or not np.all(np.isfinite(positions[:, :2])):
        raise ValueError("VLA positions need at least two finite channel/depth rows")
    depths = positions[:, 1][::-1].copy()
    channels = origin/"events"/event/"vla_matfiles"
    first = load_channel(channels/f"{event}_VLA_NO_1.mat")
    signal = np.empty((len(depths), len(first)), dtype=np.float64); signal[0] = first
    paths = [positions_path, channels/f"{event}_VLA_NO_1.mat"]
    for index in range(1, len(depths)):
        path = channels/f"{event}_VLA_NO_{index+1}.mat"; channel = load_channel(path)
        if len(channel) != signal.shape[1]:
            raise ValueError(f"Channel {index+1} sample count does not match channel 1")
        signal[index] = channel; paths.append(path)
    range_path = origin/"events/range"/f"RangeEvent{event}"/f"SproulToVLA.{event}.txt"
    range_data = np.loadtxt(range_path, skiprows=1, usecols=(2, 3), ndmin=2)
    keep = np.all(np.isfinite(range_data), axis=1); range_data = range_data[keep]
    if len(range_data) < 2:
        raise ValueError("Range labels need at least two finite rows")
    order = np.argsort(range_data[:, 0], kind="stable")
    range_times, ranges = range_data[order, 0]*60, range_data[order, 1]
    if np.any(np.diff(range_times) <= 0):
        raise ValueError("Range label times must be unique and strictly increasing")
    paths.append(range_path)
    ctd_path = origin/"environments/ctds/i9605.prn"
    ctd = np.loadtxt(ctd_path, ndmin=2)
    if ctd.shape[1] < 5:
        raise ValueError("CTD PRN must contain at least five numeric columns")
    ctd = ctd[:, :5]; ctd = ctd[np.all(np.isfinite(ctd), axis=1)]
    if len(ctd) < 2 or np.any(np.diff(ctd[:, 0]) <= 0) or np.any(ctd[:, 3] <= 0):
        raise ValueError("CTD profile needs increasing finite depths and positive sound speeds")
    profile_depths, speeds = extend_profile(ctd[:, 0], ctd[:, 3], np.max(depths)); paths.append(ctd_path)
    return {"signal": signal, "depths": depths, "range_times": range_times, "ranges": ranges,
            "range_path": str(range_path), "ctd_path": str(ctd_path),
            "profile_depths": profile_depths, "sound_speeds": speeds,
            "input_files": [_input_file_identity(path) for path in paths]}


def centered_window(signal, start, stop):
    """Center each channel using only this window; leave raw observations intact."""
    section = signal[:, start:stop]
    return section - np.mean(section, axis=1, keepdims=True)


def generation_identity(config):
    return _generation_identity_common(config, generator_version=GENERATOR_VERSION)


def dataset_names(method, config, modes, split, windows):
    return _dataset_names_common(method, config, modes, split, windows, generator_version=GENERATOR_VERSION)


def create_h5(path, method, indices, windows, inputs, frequencies, feature_size, pair_i=None, pair_j=None, theta=None):
    return _create_h5_common(path, method, indices, windows, inputs, frequencies, feature_size,
                             pair_i, pair_j, theta, generator_version=GENERATOR_VERSION)


def generate_method(method, config, inputs, base_windows, root=ROOT):
    return _generate_method_common(method, config, inputs, base_windows, root,
                                   generator_version=GENERATOR_VERSION,
                                   window_section_fn=centered_window)


def generate_datasets(raw_config, root=ROOT):
    return _generate_datasets(
        raw_config,
        root,
        normalize_config_fn=normalize_config,
        load_inputs_fn=load_inputs,
        methods=("RBD", "SCM"),
        generator_version=GENERATOR_VERSION,
        signal_centering="per_element_per_window_mean",
        window_section_fn=centered_window,
        extra_manifest_dirs=(PACK_DIR,),
    )


__all__ = [
    "GENERATOR_VERSION", "ROOT", "centered_window", "create_h5", "dataset_names",
    "generate_datasets", "generate_method", "generation_identity", "json_value",
    "load_channel", "load_inputs", "output_path", "source_manifest", "write_json",
]
