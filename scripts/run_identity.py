"""Canonical descriptive IDs, shared by dispatch, workers and retrieval."""
from __future__ import annotations

from datetime import datetime, timezone
import re
import uuid

from scripts.packs import NETWORKS, get_pack

_METHOD = r"(?:all|RBD|ELM|SCM|RBD_ELM|RBD_SCM|ELM_SCM|RBD_ELM_SCM)"
_NETWORK = "(?:all|" + "|".join(NETWORKS) + ")"
RUN_ID = re.compile(
    rf"(?P<time>[0-9]{{8}}T[0-9]{{6}}Z)-(?P<pack>base|joint)-"
    rf"(?P<split>full|before|after)-(?P<method>{_METHOD})-(?P<network>{_NETWORK})"
    rf"(?:-(?P<fusion>block_rms|raw_concat))?-(?P<token>[a-f0-9]{{8}})"
)


def validate_filters(pack_name: str, selection: object) -> dict:
    pack = get_pack(pack_name)
    if not isinstance(selection, dict) or set(selection) != {"method", "network"}:
        raise ValueError("Selection must contain exactly method and network filters")
    method, network = selection["method"], selection["network"]
    if method is not None and (not isinstance(method, str) or method not in pack.methods):
        raise ValueError("Invalid method filter")
    if network is not None and (not isinstance(network, str) or network not in NETWORKS):
        raise ValueError("Invalid network filter")
    if method is None and network is None:
        raise ValueError("A selected experiment requires a method or network filter")
    return dict(selection)


def identity_fields(pack_name: str, config: dict, selection: dict | None = None) -> dict:
    get_pack(pack_name)
    selected = validate_filters(pack_name, selection) if selection is not None else {"method": None, "network": None}
    dataset = config["dataset"]
    strategy = dataset["split_strategy"]
    if strategy not in {"full_interleaved", "range_overlap"}:
        raise ValueError("Invalid split strategy for run ID")
    split = "full" if strategy == "full_interleaved" else dataset["train_side"]
    if split not in {"full", "before", "after"}:
        raise ValueError("Invalid split side for run ID")
    fusion = config["fusion"]["mode"] if pack_name == "joint_feature_pack" else None
    return {"pack": "base" if pack_name == "base_pack" else "joint", "split": split,
            "method": selected["method"] or "all", "network": selected["network"] or "all", "fusion": fusion}


def parse_run_id(run_id: str, pack_name: str | None = None) -> dict:
    match = RUN_ID.fullmatch(run_id) if isinstance(run_id, str) else None
    if match is None:
        raise ValueError("Invalid run ID; use the descriptive ID printed at launch")
    fields = match.groupdict()
    datetime.strptime(fields["time"], "%Y%m%dT%H%M%SZ")
    actual_pack = "base_pack" if fields["pack"] == "base" else "joint_feature_pack"
    pack = get_pack(actual_pack)
    if pack_name is not None and actual_pack != pack_name:
        raise ValueError("Run ID pack mismatch")
    if fields["method"] != "all" and fields["method"] not in pack.methods:
        raise ValueError("Run ID method does not belong to pack")
    if (pack.fusion_modes and fields["fusion"] not in pack.fusion_modes) or (not pack.fusion_modes and fields["fusion"] is not None):
        raise ValueError("Run ID fusion does not belong to pack")
    return fields


def make_run_id(pack_name: str, config: dict, selection: dict | None = None) -> str:
    fields = identity_fields(pack_name, config, selection)
    parts = [datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
             fields["pack"], fields["split"], fields["method"], fields["network"]]
    if fields["fusion"] is not None:
        parts.append(fields["fusion"])
    run_id = "-".join([*parts, uuid.uuid4().hex[:8]])
    parse_run_id(run_id, pack_name)
    return run_id


def validate_run_identity(run_id: str, pack_name: str, config: dict, selection: dict | None = None) -> None:
    fields = parse_run_id(run_id, pack_name)
    expected = identity_fields(pack_name, config, selection)
    if any(fields[key] != value for key, value in expected.items()):
        raise ValueError("Run ID parameters differ from the frozen request")
