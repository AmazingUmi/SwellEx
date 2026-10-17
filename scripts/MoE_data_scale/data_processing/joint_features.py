"""Materialize aligned feature combinations without fitting preprocessing scales."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import h5py
import numpy as np

from scripts.MoE_data_scale.localization_networks.common.input_contract import read_input_signature
from scripts.MoE_data_scale.localization_networks.rbd_scm.contract import (
    FORMAT_VERSION, METHODS, components, validate_signature,
)

ROUTES = ("RBD", "SCM")
ALIGNMENT_FIELDS = ("y_range_km", "valid_sample", "split/source_segment_idx",
                    "time/window_start_s", "time/window_center_s", "time/window_stop_s",
                    "time/sample_start_idx", "time/sample_stop_idx", "frequency/freq_hz",
                    "array/element_idx", "array/depth_m", "range/range_time_s", "range/range_km_raw")


def _only(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {pattern} in {directory}, found {len(matches)}.")
    return matches[0]


def _source_contract(metadata: dict) -> dict:
    config = dict(metadata["effective_config"])
    config.pop("result_path", None)  # Output location is not a physical input identity.
    return {"input_files": metadata["input_files"], "effective_config": config,
            "split": metadata["split"], "generator_version": metadata["generator_version"]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_joint_dataset(dataset_dirs: dict[str, str | Path], output_dir: Path) -> dict[str, str]:
    """Build the RBD_SCM input from native, already validated route datasets.

    Joint HDF5 stores raw blocks. RMS is fitted later, after the neural
    train/validation split and retained optimizer-training subset.
    The complete output directory is published atomically; existing outputs
    are refused rather than silently reused.
    """
    fusion_strategy = "block_rms"
    if not set(ROUTES).issubset(dataset_dirs) or set(dataset_dirs) - {"RBD", "ELM", "SCM"}:
        raise ValueError("Joint generation requires RBD and SCM route directories.")
    directories = {method: Path(path).resolve() for method, path in dataset_dirs.items() if method in ROUTES}
    metadata = {method: json.loads(_only(path, "*_metadata.json").read_text(encoding="utf-8"))
                for method, path in directories.items()}
    source = _source_contract(metadata["RBD"])
    for method in ROUTES:
        if _source_contract(metadata[method]) != source:
            raise ValueError(f"{method} raw provenance/configuration/split differs from RBD.")
    cfg = source["effective_config"]["dataset"]
    if cfg["num_snapshots_per_segment"] != 1 or cfg["snapshot_overlap_count"] != 0:
        raise ValueError("Joint features require Ns=1 and no snapshot overlap.")
    files = {split: {method: _only(directories[method], f"*_{split}.h5") for method in ROUTES}
             for split in ("train", "test")}
    signatures = {}
    split_ids = {}
    for split, paths in files.items():
        with ExitStack() as stack:
            handles = {method: stack.enter_context(h5py.File(path, "r")) for method, path in paths.items()}
            first = handles["RBD"]
            for method, handle in handles.items():
                shape = tuple(handle["X"].shape)
                order = f"sample,{'element' if method == 'RBD' else 'pair'},frequency,real_imag"
                if (len(shape) != 4 or shape[-1] != 2 or handle["X"].attrs.get("axis_order") != order
                        or handle["X"].attrs.get("storage_axis_order") != order):
                    raise ValueError("Joint construction requires native, explicitly declared feature axes.")
                if handle.attrs.get("generator_version") != source["generator_version"]:
                    raise ValueError("Source HDF5/metadata generator versions differ.")
                signature = read_input_signature(handle, (2, shape[1], shape[2]), method)
                if method in signatures and signatures[method] != signature:
                    raise ValueError(f"{method} train/test physical axes differ.")
                signatures[method] = signature
                for field in ALIGNMENT_FIELDS:
                    if field not in handle or field not in first or not np.array_equal(handle[field][:], first[field][:]):
                        raise ValueError(f"{method} {split} alignment mismatch: {field}.")
                labels = np.asarray(handle["y_range_km"]).reshape(-1)
                valid = np.asarray(handle["valid_sample"]).reshape(-1)
                if shape[0] != labels.size or valid.size != labels.size or not np.isfinite(labels).all() or not np.all(valid == 1):
                    raise ValueError("Joint inputs must contain exactly the aligned valid finite rows.")
                for field in ("split/source_segment_idx", "time/window_start_s", "time/window_center_s",
                              "time/window_stop_s", "time/sample_start_idx", "time/sample_stop_idx"):
                    values = np.asarray(handle[field])
                    if (values.ndim not in (1, 2) or (values.ndim == 2 and 1 not in values.shape)
                            or values.size != shape[0] or values.dtype.kind not in "iuf"
                            or not np.isfinite(values).all()):
                        raise ValueError(f"Joint sample metadata must be finite vectors matching X: {field}.")
            source_ids = np.asarray(first["split/source_segment_idx"]).reshape(-1)
            if (not np.isfinite(source_ids).all() or (source_ids < 1).any()
                    or not np.equal(source_ids, np.floor(source_ids)).all()
                    or len(np.unique(source_ids)) != len(source_ids)):
                raise ValueError("Source segment identities must be unique positive integers.")
            split_ids[split] = set(source_ids.tolist())
    if split_ids["train"] & split_ids["test"]:
        raise ValueError("Joint train/test source identities overlap.")
    fingerprints = {split: {method: {"path": str(path), "sha256": _sha256(path)}
                           for method, path in paths.items()} for split, paths in files.items()}
    digest = hashlib.sha256(json.dumps({"version": FORMAT_VERSION, "sources": fingerprints,
                                       "fusion_strategy": fusion_strategy}, sort_keys=True).encode()).hexdigest()[:12]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for method in METHODS:
        blocks = []
        height = 0
        for route in components(method):
            count = signatures[route]["input_shape"][1]
            blocks.append({"method": route, "start": height, "stop": height + count,
                           "input_signature": signatures[route]})
            height += count
        shape = (2, height, signatures["RBD"]["input_shape"][2])
        signature = {"format_version": FORMAT_VERSION, "method": method, "input_shape": list(shape),
                     "fusion_strategy": fusion_strategy, "blocks": blocks, "source_contract": source}
        validate_signature(signature, shape, method)
        destination = output_root / f"{method}_{fusion_strategy}_{digest}"
        if destination.exists():
            raise FileExistsError(f"Joint dataset already exists: {destination}")
        temporary = output_root / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            for split, paths in files.items():
                with ExitStack() as stack:
                    handles = {route: stack.enter_context(h5py.File(paths[route], "r")) for route in components(method)}
                    first = next(iter(handles.values()))
                    target = stack.enter_context(h5py.File(temporary / f"{method}_{split}.h5", "x"))
                    count = first["X"].shape[0]
                    x = target.create_dataset("X", shape=(count, height, shape[2], 2), dtype="float32",
                                              chunks=(1, height, shape[2], 2), compression="gzip", compression_opts=4)
                    x.attrs["axis_order"] = x.attrs["storage_axis_order"] = "sample,feature,frequency,real_imag"
                    target.attrs["pack"] = "MoE_data_scale"
                    target.attrs["fusion_input_signature"] = json.dumps(signature, sort_keys=True, allow_nan=False)
                    for field in ALIGNMENT_FIELDS:
                        # Array/frequency identity belongs inside each block, not to a fake combined pair axis.
                        if field.startswith(("frequency/", "array/", "range/")):
                            continue
                        target.create_dataset(field, data=first[field][:])
                    for block in blocks:
                        source_x = handles[block["method"]]["X"]
                        for start in range(0, count, 64):
                            values = source_x[start:start + 64]
                            if not np.isfinite(values).all():
                                raise ValueError("Joint source contains nonfinite features.")
                            x[start:start + 64, block["start"]:block["stop"], :, :] = values
            (temporary / f"{method}_metadata.json").write_text(
                json.dumps({"pack": "MoE_data_scale", "input_signature": signature,
                            "source_h5": fingerprints}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            os.rename(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary)
            raise
        results[method] = str(destination.resolve())
    return results
