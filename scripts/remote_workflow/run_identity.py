"""Canonical descriptive IDs, shared by dispatch, workers and retrieval."""
from __future__ import annotations

from datetime import datetime, timezone
import re
import uuid

from scripts.data_processing.input_contract import METHODS
from scripts.remote_workflow.packs import NETWORKS, PACKS, get_pack

_METHOD_ITEM = "(?:" + "|".join(re.escape(m) for m in METHODS) + ")"
_NETWORK_ITEM = "(?:" + "|".join(re.escape(n) for n in NETWORKS) + ")"
_METHOD = f"(?:all|{_METHOD_ITEM}(?:\\+{_METHOD_ITEM})*)"
_NETWORK = f"(?:all|{_NETWORK_ITEM}(?:\\+{_NETWORK_ITEM})*)"
_PACK = "|".join(re.escape(pack.tag) for pack in PACKS.values())
RUN_ID = re.compile(
    rf"(?P<time>[0-9]{{8}}T[0-9]{{6}}Z)-(?P<pack>{_PACK})-"
    rf"(?P<split>full|before|after)-(?P<method>{_METHOD})-(?P<network>{_NETWORK})"
    rf"-(?P<token>[a-f0-9]{{8}})"
)


def _encode_subset(items: tuple[str, ...] | list[str], universe: tuple[str, ...]) -> str:
    chosen = set(items)
    if chosen == set(universe):
        return "all"
    ordered = [item for item in universe if item in chosen]
    if not ordered:
        raise ValueError("Subset cannot be empty")
    return "+".join(ordered)


def _validate_subset_token(token: str, universe: tuple[str, ...], kind: str) -> None:
    if token == "all":
        return
    parts = token.split("+")
    if any(part not in universe for part in parts):
        raise ValueError(f"Run ID {kind} does not belong to pack")
    if len(set(parts)) != len(parts):
        raise ValueError(f"Run ID {kind} contains duplicate entries")
    if len(parts) >= len(universe):
        raise ValueError(f"Run ID {kind} covering all entries must use 'all'")
    expected = [item for item in universe if item in set(parts)]
    if parts != expected:
        raise ValueError(f"Run ID {kind} entries are not in canonical pack order")


def validate_filters(pack_name: str, selection: object) -> dict:
    pack = get_pack(pack_name)
    if not isinstance(selection, dict):
        raise ValueError("Selection must be a dictionary")
    if set(selection) == {"methods", "networks"}:
        methods, networks = selection["methods"], selection["networks"]
        if (
            not isinstance(methods, list)
            or not methods
            or any(not isinstance(m, str) or m not in pack.methods for m in methods)
            or len(set(methods)) != len(methods)
        ):
            raise ValueError("Invalid methods selection list")
        if (
            not isinstance(networks, list)
            or not networks
            or any(not isinstance(n, str) or n not in pack.networks for n in networks)
            or len(set(networks)) != len(networks)
        ):
            raise ValueError("Invalid networks selection list")
        return {"methods": list(methods), "networks": list(networks)}
    if set(selection) == {"method", "network"}:
        method, network = selection["method"], selection["network"]
        if method is not None and (not isinstance(method, str) or method not in pack.methods):
            raise ValueError("Invalid method filter")
        if network is not None and (not isinstance(network, str) or network not in pack.networks):
            raise ValueError("Invalid network filter")
        if method is None and network is None:
            raise ValueError("A selected experiment requires a method or network filter")
        return dict(selection)
    raise ValueError("Selection must contain either {methods, networks} or {method, network}")


def resolve_selection_lists(pack_name: str, selection: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    pack = get_pack(pack_name)
    validated = validate_filters(pack_name, selection)
    if "methods" in validated:
        method_set = set(validated["methods"])
        network_set = set(validated["networks"])
        methods = tuple(m for m in pack.methods if m in method_set)
        networks = tuple(n for n in pack.networks if n in network_set)
        return methods, networks
    methods = (validated["method"],) if validated["method"] is not None else pack.methods
    networks = (validated["network"],) if validated["network"] is not None else pack.networks
    return methods, networks


def identity_fields(pack_name: str, config: dict, selection: dict | None = None) -> dict:
    pack = get_pack(pack_name)
    if selection is None:
        methods, networks = pack.methods, pack.networks
    else:
        methods, networks = resolve_selection_lists(pack_name, selection)
    dataset = config["dataset"]
    strategy = dataset["split_strategy"]
    if strategy not in {"full_interleaved", "range_overlap"}:
        raise ValueError("Invalid split strategy for run ID")
    split = "full" if strategy == "full_interleaved" else dataset["train_side"]
    if split not in {"full", "before", "after"}:
        raise ValueError("Invalid split side for run ID")
    return {
        "pack": pack.tag,
        "split": split,
        "method": _encode_subset(methods, pack.methods),
        "network": _encode_subset(networks, pack.networks),
    }


def parse_run_id(run_id: str, pack_name: str | None = None) -> dict:
    match = RUN_ID.fullmatch(run_id) if isinstance(run_id, str) else None
    if match is None:
        raise ValueError("Invalid run ID; use the descriptive ID printed at launch")
    fields = match.groupdict()
    datetime.strptime(fields["time"], "%Y%m%dT%H%M%SZ")
    actual_pack = next(pack.name for pack in PACKS.values() if pack.tag == fields["pack"])
    pack = get_pack(actual_pack)
    if pack_name is not None and actual_pack != pack_name:
        raise ValueError("Run ID pack mismatch")
    _validate_subset_token(fields["method"], pack.methods, "method")
    _validate_subset_token(fields["network"], pack.networks, "network")
    return fields


def make_run_id(pack_name: str, config: dict, selection: dict | None = None) -> str:
    fields = identity_fields(pack_name, config, selection)
    parts = [
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        fields["pack"],
        fields["split"],
        fields["method"],
        fields["network"],
    ]
    run_id = "-".join([*parts, uuid.uuid4().hex[:8]])
    parse_run_id(run_id, pack_name)
    return run_id


def validate_run_identity(
    run_id: str,
    pack_name: str,
    config: dict,
    selection: dict | None = None,
) -> None:
    fields = parse_run_id(run_id, pack_name)
    expected = identity_fields(pack_name, config, selection)
    if any(fields[key] != value for key, value in expected.items()):
        raise ValueError("Run ID parameters differ from the frozen request")
