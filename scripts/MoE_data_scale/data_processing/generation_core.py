"""Trusted numeric input loading, logical-axis HDF5 writing, and orchestration."""
from __future__ import annotations

import hashlib
import json
import platform
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import scipy
from scipy.io import loadmat

from scripts.MoE_data_scale.data_processing.feature_math import (
    segments, combine_snapshots, build_split, pair_indices, select_frequency_bins,
    extend_profile, steering_delays, sidelobe_psf, rbd_decompose,
    elm_feature, scm_feature, round_positive,
)
from scripts.MoE_data_scale.data_processing.generation_config import ROOT, normalize_config, output_path

GENERATOR_VERSION = "python-features-v2"


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path, value, root=ROOT):
    target = output_path(str(path), root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(target)


def source_manifest():
    directory = Path(__file__).resolve().parent
    return [{"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(directory.glob("*.py"))]


def _real_numeric_vector(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or np.iscomplexobj(array) or array.ndim > 2 or (array.ndim == 2 and min(array.shape) != 1):
        raise ValueError(f"{name} must be a real numeric vector; objects, structs, and references are not accepted")
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if len(array) < 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain at least two finite numeric samples")
    return array


def load_channel(path):
    """Read only the trusted numeric x variable, never Python serialized objects."""
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            if "x" not in handle or not isinstance(handle["x"], h5py.Dataset):
                raise ValueError(f"Missing numeric x in {path}")
            variable = handle["x"]
            if h5py.check_dtype(ref=variable.dtype) is not None or variable.dtype.kind not in "iuf":
                raise ValueError(f"Unsupported nonnumeric/reference x in {path}")
            return _real_numeric_vector(variable[...], str(path))
    contents = loadmat(path, variable_names=["x"], simplify_cells=True)
    if "x" not in contents:
        raise ValueError(f"Missing x in {path}")
    return _real_numeric_vector(contents["x"], str(path))


def _input_file_identity(path):
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    current = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
        raise ValueError(f"Input file changed while hashing: {path}")
    return {"path": str(path), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def load_inputs(root, config):
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
    # Preserve DS_load_vla_signals' full-record per-element mean centering.
    signal -= np.mean(signal, axis=1, keepdims=True)
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
    # read_ctd_i9605.m constructs its opaque MATLAB table from these same five
    # columns. Reading the original numeric PRN preserves that source without
    # depending on MATLAB table deserialization.
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


def _seconds_tag(value):
    return f"{value:.12g}".replace(".", "p")


def generation_identity(config):
    """Hash normalized computation settings, independent of output/report locations."""
    payload = {"generator_version": GENERATOR_VERSION, "schema_version": config["schema_version"],
               "dataset": config["dataset"], "rbd": config["rbd"]}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def dataset_names(method, config, modes, split, windows):
    d, r = config["dataset"], config["rbd"]
    frequency = "_".join(mode+str(d["frequency_selection_config"].get(mode+"_num_bins", 64 if mode == "mel" else 16))
                         if mode in {"mel", "adapt"} else mode for mode in modes)
    if method == "RBD":
        variant = (f"rbd_green_{frequency}_est{'full' if r['frequency_estimation']=='full' else 'sel'}"
                   f"_seg{_seconds_tag(d['segment_duration_s'])}s_step{_seconds_tag(d['segment_step_s'])}s"
                   f"_norm{int(r['normalize_spectrum'])}_pw{int(r['use_plane_wave'])}_"
                   f"{'bestbeam' if r['beam_selection']=='best' else 'multipath'}")
    else:
        feature = "elm_pairwise_lsr_upper" if method == "ELM" else "scm_upper_diag"
        variant = (f"{feature}_{frequency}_snap{_seconds_tag(d['segment_duration_s'])}s"
                   f"_ns{d['num_snapshots_per_segment']}_ov{d['snapshot_overlap_count']}")
    if d["manual_dataset_variant_tag"]:
        variant += "_"+d["manual_dataset_variant_tag"]
    if d["split_strategy"] == "full_interleaved":
        split_name = f"full_interleaved_{d['train_test_ratio'][0]}_{d['train_test_ratio'][1]}"
        time_tag = (f"s{int(round_positive(windows['candidate_start_s']))}_to_s{int(round_positive(windows['candidate_end_s']))}"
                    f"_step_s{int(round_positive(windows['step_s']))}")
    else:
        split_name = f"range_overlap_train_{d['train_side']}_{int(round_positive(d['half_duration_s']))}s_gap{int(round_positive(d['gap_s']))}s"
        if d["test_step_s"] is not None:
            split_name += f"_teststep{int(round_positive(d['test_step_s']))}s"
        low = min(split["left_time_start_s"], split["right_time_start_s"])
        high = max(split["left_time_end_s"], split["right_time_end_s"])
        time_tag = (f"selected_s{int(round_positive(low))}_to_s{int(round_positive(high))}"
                    f"_min_s{int(round_positive(split['range_min_time_s']))}_step_s{int(round_positive(windows['step_s']))}")
    name = split_name+"_"+variant+"_"+generation_identity(config)
    prefix = {"RBD": "RBD_green_freq_nn", "ELM": "ELM_lsr_freq_nn", "SCM": "SCM_freq_nn"}[method]
    return name, variant, f"{prefix}_{d['event_name']}_{time_tag}_{variant}"


def _column(handle, name, value, dtype):
    handle.create_dataset(name, data=np.asarray(value, dtype=dtype).reshape(-1, 1))


def _create_h5(path, method, indices, windows, inputs, frequencies, feature_size, pair_i=None, pair_j=None, theta=None):
    handle = h5py.File(path, "x")
    try:
        x = handle.create_dataset("X", shape=(len(indices), feature_size, len(frequencies), 2), dtype="float32",
                                  chunks=(1, feature_size, len(frequencies), 2), compression="gzip", compression_opts=4)
        axis_order = f"sample,{'element' if method=='RBD' else 'pair'},frequency,real_imag"
        x.attrs["axis_order"] = axis_order; x.attrs["storage_axis_order"] = axis_order
        x.attrs["feature_definition"] = {"RBD": "rbd_result.green_freq", "ELM": "least_squares_ratio_across_snapshots",
                                         "SCM": "mean_normalized_snapshot_outer_products"}[method]
        handle.attrs["generator_version"] = GENERATOR_VERSION
        _column(handle, "y_range_km", windows["labels"][indices], "float32")
        _column(handle, "valid_sample", windows["valid"][indices], "uint8")
        for key, source in (("window_start_s", "start"), ("window_center_s", "center"), ("window_stop_s", "stop")):
            _column(handle, "time/"+key, windows[source][indices], "float64")
        _column(handle, "time/sample_start_idx", windows["sample_start"][indices]+1, "uint64")
        _column(handle, "time/sample_stop_idx", windows["sample_stop"][indices], "uint64")
        _column(handle, "frequency/freq_hz", frequencies, "float64")
        _column(handle, "array/depth_m", inputs["depths"], "float64")
        _column(handle, "array/element_idx", np.arange(1, len(inputs["depths"])+1), "uint16")
        _column(handle, "range/range_time_s", inputs["range_times"], "float64")
        _column(handle, "range/range_km_raw", inputs["ranges"], "float64")
        _column(handle, "split/source_segment_idx", indices+1, "uint64")
        if pair_i is not None:
            _column(handle, "pair/numerator_element_idx", pair_i+1, "uint16")
            _column(handle, "pair/denominator_element_idx", pair_j+1, "uint16")
        if theta is not None:
            _column(handle, "rbd/theta_vec_rad", theta, "float64")
            for name, shape, dtype in (("theta_best_rad", (len(indices), 1), "float64"),
                                       ("num_selected_angles", (len(indices), 1), "uint16"),
                                       ("theta_selected_rad", (len(indices), len(theta)), "float64"),
                                       ("selected_beam_power", (len(indices), len(theta)), "float32"),
                                       ("beam_power", (len(indices), len(theta)), "float32"),
                                       ("signal_freq_scale", (len(indices), 1), "float64")):
                handle.create_dataset("rbd/"+name, shape=shape, dtype=dtype)
        return handle
    except BaseException:
        handle.close(); raise


@contextmanager
def create_h5(path, method, indices, windows, inputs, frequencies, feature_size, pair_i=None, pair_j=None, theta=None):
    """Publish a closed HDF5 file atomically and never replace an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Dataset already exists: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with _create_h5(temporary, method, indices, windows, inputs, frequencies, feature_size, pair_i, pair_j, theta) as handle:
            yield handle
        # An exclusive hard link publishes the completed file atomically; unlike
        # replace(), it also refuses an output created by a concurrent writer.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rbd_diagnostics(handle, local, result, slots):
    handle["rbd/theta_best_rad"][local, 0] = result["theta_best"]
    handle["rbd/num_selected_angles"][local, 0] = result["num_selected_angles"]
    order = np.argsort(-result["selected_beam_power"], kind="stable")[:slots]
    theta_row = np.full(slots, np.nan); power_row = np.full(slots, np.nan)
    theta_row[:len(order)] = result["theta_selected"][order]
    power_row[:len(order)] = result["selected_beam_power"][order]
    handle["rbd/theta_selected_rad"][local] = theta_row
    handle["rbd/selected_beam_power"][local] = power_row
    handle["rbd/beam_power"][local] = result["beam_power"]
    handle["rbd/signal_freq_scale"][local, 0] = result["signal_freq_scale"]


def generate_method(method, config, inputs, base_windows, root=ROOT):
    d, r = config["dataset"], config["rbd"]
    windows = base_windows if method == "RBD" else combine_snapshots(base_windows, d["num_snapshots_per_segment"],
                  d["snapshot_overlap_count"], inputs["range_times"], inputs["ranges"])
    splits, split_metadata = build_split(windows, d)
    full_frequencies = np.arange(windows["snapshot_num_samples"]//2+1)*d["fs"]/windows["snapshot_num_samples"]
    modes = ["full"] if method == "RBD" and r["frequency_estimation"] == "full" else d["frequency_selection_modes"]
    bins, frequencies, frequency_info = select_frequency_bins(full_frequencies, modes, d["frequency_selection_config"],
                                  signal=inputs["signal"], windows=windows, train_indices=splits["train"])
    name, variant, stem = dataset_names(method, config, modes, split_metadata, windows)
    directory = Path(root)/"outputs/Datasets"/name; directory.mkdir(parents=True, exist_ok=False)
    theta = np.deg2rad(np.linspace(-90, 90, 181)) if method == "RBD" else None
    if method == "RBD":
        pair_i = pair_j = None; size = len(inputs["depths"])
        delays = steering_delays(theta, inputs["depths"], inputs["sound_speeds"], inputs["profile_depths"], r["use_plane_wave"])
        psf = sidelobe_psf(delays, 2*np.pi*frequencies) if r["beam_selection"] == "multipath" else None
    else:
        pair_i, pair_j = pair_indices(len(inputs["depths"]), method == "SCM"); size = len(pair_i)
    paths = {split: directory/f"{stem}_{split}.h5" for split in splits}
    for split, indices in splits.items():
        with create_h5(paths[split], method, indices, windows, inputs, frequencies, size, pair_i, pair_j, theta) as handle:
            for local, index in enumerate(indices):
                section = inputs["signal"][:, windows["sample_start"][index]:windows["sample_stop"][index]]
                if method == "RBD":
                    _, returned_frequencies, result = rbd_decompose(section, d["fs"], theta, delays, bins=bins,
                          normalize_spectrum=r["normalize_spectrum"], multipath=r["beam_selection"] == "multipath",
                          multipath_options=r["multipath_options"], psf=psf)
                    if not np.array_equal(returned_frequencies, frequencies):
                        raise ValueError("RBD frequency axis does not match frozen frequency selection")
                    feature = result["green_freq"]
                    _rbd_diagnostics(handle, local, result, len(theta))
                elif method == "ELM":
                    feature, _ = elm_feature(section, windows["snapshot_num_samples"], windows["snapshot_step_samples"], bins, pair_i, pair_j)
                else:
                    feature, _ = scm_feature(section, windows["snapshot_num_samples"], windows["snapshot_step_samples"], bins, pair_i, pair_j)
                if not np.all(np.isfinite(feature)):
                    raise ValueError(f"Nonfinite {method} feature at source window {index+1}")
                handle["X"][local, :, :, 0] = np.real(feature).astype(np.float32)
                handle["X"][local, :, :, 1] = np.imag(feature).astype(np.float32)
        print(f"{method} {split}: {len(indices)} windows -> {paths[split]}", flush=True)
    metadata = {"format": f"Neural-network dataset: X(window, {'element' if method=='RBD' else 'pair'}, frequency, real_imag), y_range_km(window).",
                "real_imag_index": "X[...,0]=real(feature), X[...,1]=imag(feature).", "generator_version": GENERATOR_VERSION,
                "axis_order": f"sample,{'element' if method=='RBD' else 'pair'},frequency,real_imag",
                "split_strategy": d["split_strategy"], "split_strategy_dir_name": name, "dataset_variant_tag": variant,
                "global_num_segments": len(windows["start"]), "candidate_segment_start_s": windows["candidate_start_s"],
                "candidate_segment_end_s": windows["candidate_end_s"], "fs_hz": d["fs"], "segment_duration_s": windows["duration_s"],
                "segment_step_s": windows["step_s"], "range_file": inputs["range_path"], "ctd_file": inputs["ctd_path"],
                "freq_num": len(frequencies), "freq_min_hz": float(frequencies[0]), "freq_max_hz": float(frequencies[-1]),
                "num_elements": len(inputs["depths"]), "split": split_metadata,
                "input_files": inputs["input_files"],
                "files": [{"name": split, "file": str(path)} for split, path in paths.items()], "effective_config": config}
    feature_config = {"frequency_selection": "combined_frequency_selection_modes", "frequency_selection_modes": modes,
                      "selected_fft_bin_idx": bins+1, "selected_freq_hz": frequencies, "frequency_selection_info": frequency_info,
                      "num_freq_bins": len(frequencies), "manual_dataset_variant_tag": d["manual_dataset_variant_tag"]}
    if method == "RBD":
        feature_config.update(normalize_spectrum=r["normalize_spectrum"], use_plane_wave=r["use_plane_wave"],
                       rbd_beam_selection=r["beam_selection"], multipath_beam=r["beam_selection"] == "multipath",
                       rbd_frequency_estimation=r["frequency_estimation"], rbd_selected_frequency_modes=modes,
                       rbd_multipath_options=r["multipath_options"], num_arrival_slots=len(theta),
                       full_num_freq_bins=len(full_frequencies), output_definition="rbd_result.green_freq")
        metadata.update(rbd_config=feature_config, normalize_spectrum=r["normalize_spectrum"], use_plane_wave=r["use_plane_wave"],
                        theta_num=len(theta), theta_min_rad=float(theta[0]), theta_max_rad=float(theta[-1]))
    else:
        feature_config.update(mode="element_pairwise_frequency_ratio" if method == "ELM" else "spatial_covariance_matrix",
                   definition="ratio=sum_s Xi*conj(Xj)/(sum_s abs(Xj)^2+floor+eps)" if method == "ELM" else "C=mean_s((x/||x||)(x/||x||)^H)",
                   num_pairs=len(pair_i), pair_selection="strict_upper_triangle_i_lt_j" if method == "ELM" else "upper_triangle_i_le_j",
                   pair_numerator_element_idx=pair_i+1, pair_denominator_element_idx=pair_j+1,
                   snapshot_duration_s=d["segment_duration_s"], snapshot_step_s=d["segment_step_s"],
                   num_snapshots_per_segment=d["num_snapshots_per_segment"], snapshot_overlap_count=d["snapshot_overlap_count"],
                   snapshot_num_samples=windows["snapshot_num_samples"], snapshot_step_samples=windows["snapshot_step_samples"],
                   actual_num_snapshots=d["num_snapshots_per_segment"],
                   denom_floor_relative=1e-6 if method == "ELM" else None, norm_floor=1e-12 if method == "SCM" else None)
        metadata["feature_config"] = feature_config
    metadata_path = directory/f"{stem}_metadata.json"; write_json(metadata_path, metadata, root)
    return {"method": method, "dataset_name": name, "dataset_dir": str(directory), "train_file": str(paths["train"]),
            "test_file": str(paths["test"]), "metadata_file": str(metadata_path), "train_count": len(splits["train"]),
            "test_count": len(splits["test"]), "frequency_hz": frequencies, "frequency_selection": frequency_info,
            "split": split_metadata, "effective_config": config, "effective_segment_start_s": windows["candidate_start_s"],
            "effective_segment_end_s": windows["candidate_end_s"]}


def generate_datasets(raw_config, root=ROOT):
    root = Path(root).resolve(); config = normalize_config(raw_config, root)
    report = {"schema_version": 1, "status": "RUNNING", "generator_version": GENERATOR_VERSION,
              "project_dir": str(root), "python_version": platform.python_version(),
              "dependencies": {"numpy": np.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__},
              "config": config, "datasets": {}, "metadata": {}, "dataset_results": {}, "source_manifest": source_manifest(),
              "provenance": {"frequency_fit_scope": "training_split_only", "split_train_side": config["dataset"]["train_side"],
                             "random_numbers_used": False, "signal_centering": "per_element_full_record_mean",
                             "ctd_source": "original i9605.prn numeric depth/sound_speed columns", "rbd_output": "actual Green estimate"}}
    try:
        inputs = load_inputs(root, config); report["input_files"] = inputs["input_files"]
        d = config["dataset"]
        base = segments(inputs["signal"], d["fs"], d["segment_duration_s"], d["segment_step_s"], d["segment_start_s"],
                        d["segment_end_s"], inputs["range_times"], inputs["ranges"])
        for method in ("RBD", "ELM", "SCM"):
            result = generate_method(method, config, inputs, base, root)
            report["dataset_results"][method] = result; report["datasets"][method] = result["dataset_dir"]
            report["metadata"][method] = result["metadata_file"]
        report["status"] = "PASS"
    except Exception as failure:
        report["status"] = "FAIL"; report["error"] = {"type": type(failure).__name__, "message": str(failure)}
        write_json(config["result_path"], report, root); raise
    write_json(config["result_path"], report, root)
    return report
