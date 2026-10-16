"""Canonical, strict configuration for deterministic feature generation."""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASET_DEFAULTS = {
    "event_name": "S5", "fs": 1500, "segment_start_s": None, "segment_end_s": None,
    "segment_duration_s": 1.0, "segment_step_s": 1.0,
    "num_snapshots_per_segment": 1, "snapshot_overlap_count": 0,
    "frequency_selection_modes": ["deep", "shallow"], "frequency_selection_config": {},
    "split_strategy": "full_interleaved", "train_test_ratio": [4, 1], "train_side": "before",
    "half_duration_s": 800.0, "gap_s": 15.0, "test_step_s": None,
    "manual_dataset_variant_tag": "",
}
MULTIPATH_DEFAULTS = {"peak_threshold_db": -6.0, "min_separation_deg": 2.0,
                      "max_num_peaks": 181, "sidelobe_reject_db": 3.0}
RBD_DEFAULTS = {"beam_selection": "best", "frequency_estimation": "selected",
                "use_plane_wave": False, "normalize_spectrum": False,
                "multipath_options": MULTIPATH_DEFAULTS}
FREQUENCY_KEYS = {"mel_num_bins", "mel_min_freq_hz", "mel_max_freq_hz",
                  "deep_target_freq_hz", "shallow_target_freq_hz", "adapt_num_bins",
                  "adapt_min_freq_hz", "adapt_max_freq_hz", "adapt_max_num_segments"}


def _object(value, allowed, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")


def _number(value, name, minimum=0, integer=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (positive and value <= 0) or (integer and type(value) is not int):
        raise ValueError(f"Invalid numeric {name}: {value}")


def output_path(value, root=ROOT):
    if not isinstance(value, str) or not value:
        raise ValueError("Report path must be a nonempty string")
    candidate = Path(value)
    result = (candidate if candidate.is_absolute() else Path(root) / candidate).resolve()
    outputs = (Path(root) / "outputs").resolve()
    if not result.is_relative_to(outputs) or result == outputs or result.suffix != ".json":
        raise ValueError("Report path must be a .json file inside project outputs")
    return result


def normalize_config(raw, root=ROOT):
    _object(raw, {"schema_version", "dataset", "rbd", "result_path"}, "config")
    if type(raw.get("schema_version", 1)) is not int or raw.get("schema_version", 1) != 1:
        raise ValueError("Only schema_version 1 is supported")
    _object(raw.get("dataset", {}), DATASET_DEFAULTS, "dataset")
    _object(raw.get("rbd", {}), RBD_DEFAULTS, "rbd")
    d = copy.deepcopy(DATASET_DEFAULTS) | copy.deepcopy(raw.get("dataset", {}))
    r = copy.deepcopy(RBD_DEFAULTS) | copy.deepcopy(raw.get("rbd", {}))
    if not isinstance(d["event_name"], str) or not re.fullmatch(r"S\d+", d["event_name"]):
        raise ValueError("event_name must match S followed by digits")
    _number(d["fs"], "fs", integer=True, positive=True)
    for key in ("segment_duration_s", "segment_step_s", "half_duration_s"):
        _number(d[key], key, positive=True)
    _number(d["gap_s"], "gap_s")
    _number(d["num_snapshots_per_segment"], "num_snapshots_per_segment", 1, integer=True)
    _number(d["snapshot_overlap_count"], "snapshot_overlap_count", integer=True)
    if d["snapshot_overlap_count"] >= d["num_snapshots_per_segment"]:
        raise ValueError("snapshot_overlap_count must be less than num_snapshots_per_segment")
    for key in ("segment_start_s", "segment_end_s", "test_step_s"):
        if d[key] is not None:
            _number(d[key], key)
    if d["segment_start_s"] is not None and d["segment_end_s"] is not None:
        if d["segment_end_s"] < d["segment_start_s"]:
            raise ValueError("segment_end_s must be >= segment_start_s")
    for key in ("segment_duration_s", "segment_step_s"):
        count = d[key] * d["fs"]
        if abs(count - round(count)) > 1e-8:
            raise ValueError(f"{key} must align to an integer sample count")
    if d["segment_duration_s"] * d["fs"] < 2:
        raise ValueError("Snapshot must contain at least two samples")
    modes = d["frequency_selection_modes"]
    if isinstance(modes, str):
        modes = [modes]
    if (not isinstance(modes, list) or not modes or any(not isinstance(x, str) for x in modes)
            or any(x not in {"full", "mel", "deep", "shallow", "adapt"} for x in modes)
            or len(set(modes)) != len(modes) or ("full" in modes and len(modes) != 1)):
        raise ValueError("Invalid, combined-full, or duplicate frequency_selection_modes")
    if "adapt" in modes:
        raise ValueError("data_scale_pack does not support adapt: frequency fitting must use only the retained optimizer-training subset")
    d["frequency_selection_modes"] = modes
    f = d["frequency_selection_config"]
    _object(f, FREQUENCY_KEYS, "frequency_selection_config")
    for key, value in f.items():
        if key.endswith("target_freq_hz"):
            if not isinstance(value, list) or not value:
                raise ValueError(f"{key} must be a nonempty frequency list")
            for frequency in value:
                _number(frequency, key)
                if frequency > d["fs"] / 2:
                    raise ValueError(f"{key} exceeds Nyquist")
        else:
            integral = "num_bins" in key or key == "adapt_max_num_segments"
            _number(value, key, minimum=1 if integral else 0, integer=integral)
            if key.endswith("freq_hz") and value > d["fs"] / 2:
                raise ValueError(f"{key} exceeds Nyquist")
    for prefix in ("mel", "adapt"):
        lo, hi = f.get(prefix + "_min_freq_hz"), f.get(prefix + "_max_freq_hz")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError(f"{prefix} minimum frequency must be below maximum")
    if d["split_strategy"] not in {"full_interleaved", "range_overlap"}:
        raise ValueError("split_strategy must be full_interleaved or range_overlap")
    if d["train_side"] not in {"before", "after"}:
        raise ValueError("train_side must be before or after")
    ratio = d["train_test_ratio"]
    if not isinstance(ratio, list) or len(ratio) != 2:
        raise ValueError("train_test_ratio requires two positive integers")
    for value in ratio:
        _number(value, "train_test_ratio", 1, integer=True)
    combined_step = (d["num_snapshots_per_segment"] - d["snapshot_overlap_count"]) * d["segment_step_s"]
    if d["test_step_s"] is not None and d["test_step_s"] < combined_step:
        raise ValueError("test_step_s must be >= the combined ELM/SCM window step")
    suffix = d["manual_dataset_variant_tag"]
    if not isinstance(suffix, str) or (suffix and not re.fullmatch(r"[A-Za-z0-9_-]+", suffix)):
        raise ValueError("manual_dataset_variant_tag must be a safe dataset suffix")
    if r["beam_selection"] not in {"best", "multipath"}:
        raise ValueError("beam_selection must be best or multipath")
    if r["frequency_estimation"] not in {"selected", "full"}:
        raise ValueError("frequency_estimation must be selected or full")
    if r["frequency_estimation"] == "selected" and "full" in modes:
        raise ValueError("RBD full selection requires frequency_estimation full")
    for key in ("use_plane_wave", "normalize_spectrum"):
        if not isinstance(r[key], bool):
            raise ValueError(f"{key} must be a JSON boolean")
    _object(r["multipath_options"], MULTIPATH_DEFAULTS, "multipath_options")
    r["multipath_options"] = copy.deepcopy(MULTIPATH_DEFAULTS) | r["multipath_options"]
    for key, value in r["multipath_options"].items():
        minimum = 1 if key == "max_num_peaks" else 0 if key == "min_separation_deg" else -math.inf
        _number(value, key, minimum, integer=key == "max_num_peaks")
    result_path = raw.get("result_path", "outputs/validation/generation_result.json")
    output_path(result_path, root)
    return {"schema_version": 1, "dataset": d, "rbd": r, "result_path": result_path}


def read_config(path, root=ROOT):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream, object_pairs_hook=no_duplicates,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))
    return normalize_config(raw, root)
