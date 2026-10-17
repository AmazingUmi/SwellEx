"""Canonical joint feature identities; feature rows retain their block semantics."""
from __future__ import annotations

import math

METHODS = ("RBD_SCM",)
FUSION_STRATEGIES = ("block_rms",)
FORMAT_VERSION = 1


def components(method: str) -> tuple[str, ...]:
    if method not in METHODS:
        raise ValueError(f"Joint feature method must be one of {', '.join(METHODS)}.")
    return tuple(method.split("_"))


def validate_signature(signature: dict, shape: tuple[int, int, int], method: str | None = None) -> None:
    """Validate each physical sub-axis without inventing an array/pair axis."""
    from scripts.MoE_learn.localization_networks.common.input_contract import validate_saved_input_signature
    required = {"format_version", "method", "input_shape", "fusion_strategy", "blocks", "source_contract"}
    if not isinstance(signature, dict) or set(signature) != required:
        raise ValueError("Joint input_signature must contain the complete fusion contract.")
    if (type(signature["format_version"]) is not int or signature["format_version"] != FORMAT_VERSION
            or not isinstance(signature["input_shape"], list)
            or any(type(value) is not int or value <= 0 for value in signature["input_shape"])
            or signature["input_shape"] != list(shape)):
        raise ValueError("Joint input signature version/shape mismatch.")
    if len(shape) != 3 or shape[0] != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("Joint input shape must be positive [2,feature,frequency].")
    expected = components(signature["method"])
    if method is not None and signature["method"] != method:
        raise ValueError("Joint input method mismatch.")
    if signature["fusion_strategy"] not in FUSION_STRATEGIES:
        raise ValueError("Unknown fusion strategy.")
    blocks = signature["blocks"]
    if not isinstance(blocks, list) or len(blocks) != len(expected):
        raise ValueError("Joint blocks do not match the requested combination.")
    offset = 0
    common_axes = None
    for route, block in zip(expected, blocks):
        if not isinstance(block, dict) or set(block) != {"method", "start", "stop", "input_signature"} or block["method"] != route:
            raise ValueError("Joint blocks must retain canonical route order and complete axes.")
        sig = block["input_signature"]
        if (not isinstance(sig, dict) or set(sig) != {"method", "input_shape", "frequency_hz", "element_idx", "depth_m", "pair_indices"}
                or not isinstance(sig["input_shape"], list) or len(sig["input_shape"]) != 3
                or any(type(v) is not int or v <= 0 for v in sig["input_shape"])):
            raise ValueError("Joint block physical signature is incomplete.")
        count = sig["input_shape"][1]
        if (type(block["start"]) is not int or type(block["stop"]) is not int
                or block["start"] != offset or block["stop"] != offset + count):
            raise ValueError("Joint block offsets are not contiguous.")

        validate_saved_input_signature(sig, (2, count, shape[2]), route)
        axes = {key: sig[key] for key in ("frequency_hz", "element_idx", "depth_m")}
        if common_axes is not None and axes != common_axes:
            raise ValueError("Joint blocks have mismatching frequency/array axes.")
        common_axes = axes
        offset += count
    if offset != shape[1]:
        raise ValueError("Joint block heights do not sum to the input height.")
    source = signature["source_contract"]
    if not isinstance(source, dict) or set(source) != {"input_files", "effective_config", "split", "generator_version"}:
        raise ValueError("Joint source contract is incomplete.")
    config = source["effective_config"]
    if (not isinstance(config, dict) or not isinstance(config.get("dataset"), dict)
            or not {"num_snapshots_per_segment", "snapshot_overlap_count"}.issubset(config["dataset"])
            or not isinstance(source["split"], dict) or not source["split"]
            or not isinstance(source["generator_version"], str) or not source["generator_version"]):
        raise ValueError("Joint source configuration/split/generator identity is incomplete.")
    ns, overlap = (config["dataset"][key] for key in ("num_snapshots_per_segment", "snapshot_overlap_count"))
    if type(ns) is not int or type(overlap) is not int or ns != 1 or overlap != 0:
        raise ValueError("Joint features require Ns=1 and no snapshot overlap.")
    inventory = source["input_files"]
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Joint features require raw input provenance.")
    paths = []
    for entry in inventory:
        if not isinstance(entry, dict) or not {"path", "size_bytes", "mtime_ns", "sha256"}.issubset(entry):
            raise ValueError("Raw input provenance must include path, size_bytes, mtime_ns and sha256.")
        if (not isinstance(entry["path"], str) or not entry["path"]
                or type(entry["size_bytes"]) is not int or entry["size_bytes"] <= 0
                or type(entry["mtime_ns"]) is not int or entry["mtime_ns"] < 0
                or not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in entry["sha256"])):
            raise ValueError("Invalid raw input identity.")
        paths.append(entry["path"])
    if len(set(paths)) != len(paths):
        raise ValueError("Raw input provenance contains duplicate paths.")


def validate_preprocessing(state: dict, signature: dict) -> None:
    if not isinstance(state, dict) or set(state) != {"strategy", "scales", "fit_sample_count", "fit_source_identity_sha256"}:
        raise ValueError("Joint preprocessing must contain strategy, scales, fit_sample_count and fit_source_identity_sha256.")
    if state["strategy"] != signature["fusion_strategy"]:
        raise ValueError("Joint preprocessing strategy differs from dataset signature.")
    scales = state["scales"]
    if not isinstance(scales, list) or len(scales) != len(signature["blocks"]):
        raise ValueError("Joint preprocessing scale count differs from feature blocks.")
    if any(isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("Joint preprocessing scales must be finite and positive.")
    if type(state["fit_sample_count"]) is not int or state["fit_sample_count"] <= 0:
        raise ValueError("Joint preprocessing requires a positive fit sample count.")
    digest = state["fit_source_identity_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Joint preprocessing requires a retained-training source identity SHA256.")
