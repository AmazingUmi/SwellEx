"""Serial, resumable data-size matrices using the isolated haiqin dispatcher.

A child is dispatched at most once. Re-entry follows its existing run ID, and a
failed or ambiguous attempt is never silently replaced with fresh training.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from scripts.MoE_data_scale.localization_networks.workflow import remote
from scripts.MoE_data_scale.localization_networks.workflow.config import validate_config
from scripts.run_identity import make_run_id, validate_run_identity

PACK = "MoE_data_scale"
PLAN_FIELDS = {"schema_version", "base_config", "fractions", "seeds", "method", "experts", "study_id"}
CHILD_STATES = {"PLANNED", "DISPATCHING", "UNKNOWN", "INTERRUPTED", "STARTING",
                "WAITING_FOR_RESOURCE", "RUNNING", "EXPORTING", "PASS", "FAIL"}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value, *, compact: bool = True) -> str:
    arguments = {"sort_keys": True, "allow_nan": False}
    if compact:
        arguments["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(value, **arguments).encode()).hexdigest()


def read_json(path: Path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Repeated JSON field: {key}")
            result[key] = value
        return result
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_pairs)


def validate_plan(plan: object) -> dict:
    if not isinstance(plan, dict) or set(plan) != PLAN_FIELDS or type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("Study plan requires exactly schema_version=1, study_id, base_config, fractions, experts, seeds, method")
    validate_config(plan["base_config"])
    if plan["base_config"]["training"]["train_fraction"] != 1:
        raise ValueError("Study base_config.training.train_fraction must be 1; fractions controls the matrix")
    if plan["method"] != "RBD_SCM":
        raise ValueError("MoE expert study method must be RBD_SCM")
    from scripts.MoE_data_scale.localization_networks.workflow.study_worker import safe_name
    safe_name(plan["study_id"])
    experts = plan["experts"]
    if (not isinstance(experts, list) or not experts
            or any(type(x) is not int or x not in (1, 2, 4, 8) for x in experts)
            or len(set(experts)) != len(experts)):
        raise ValueError("Study experts must be unique integers from 1, 2, 4, 8")
    fractions = plan["fractions"]
    if (not isinstance(fractions, list) or not fractions
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 < x <= 1 for x in fractions)
            or len(set(fractions)) != len(fractions)):
        raise ValueError("Study fractions must be unique finite numbers in (0, 1]")
    seeds = plan["seeds"]
    if (not isinstance(seeds, list) or not seeds
            or any(type(x) is not int or not 0 <= x <= 2**32 - 1 for x in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Study seeds must be unique integers between 0 and 4294967295")
    # Reject invalid effective configurations (for example adaptive-frequency
    # fitting with a reduced training subset) before creating any study output.
    for seed in seeds:
        for fraction in fractions:
            for experts in plan["experts"]:
                child_config(plan, seed, fraction, experts)
    return copy.deepcopy(plan)


def child_config(plan: dict, seed: int, fraction: float, experts: int) -> dict:
    if type(experts) is not int or experts not in plan["experts"]:
        raise ValueError("Expert count is outside the frozen matrix")
    config = copy.deepcopy(plan["base_config"])
    config["validation"]["seed"] = seed
    config["training"]["train_fraction"] = fraction
    return validate_config(config)


def selection_for(plan: dict, experts: int) -> tuple[str, dict]:
    if type(experts) is not int or experts not in plan["experts"]:
        raise ValueError("Expert count is outside the frozen matrix")
    return "selected", {"method": plan["method"], "network": f"complex_MoE_e{experts}"}


def matrix(plan: dict) -> list[tuple[int, float, int]]:
    return [(seed, fraction, experts) for seed in plan["seeds"]
            for fraction in plan["fractions"] for experts in plan["experts"]]


def source_inventory() -> tuple[list[dict], str]:
    inventory = []
    for relative in remote.source_paths():
        path = remote.PROJECT / relative
        before = path.stat()
        sha256 = remote.sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed while freezing study: {relative}")
        inventory.append({"path": relative.as_posix(), "size": after.st_size, "sha256": sha256})
    return inventory, digest(inventory)


def make_ledger(plan: dict, directory: Path) -> dict:
    plan = validate_plan(plan)
    if directory.name != plan["study_id"]:
        raise ValueError("Study directory differs from plan study_id")
    files, source_sha256 = source_inventory()
    children = []
    from scripts.MoE_data_scale.localization_networks.workflow.tasks import resolve_selection
    for seed, fraction, experts in matrix(plan):
        config = child_config(plan, seed, fraction, experts)
        _, selection = selection_for(plan, experts)
        model = resolve_selection(plan["method"], selection["network"])["model"]
        children.append({"run_id": make_run_id(PACK, config, selection), "seed": seed,
                         "train_fraction": fraction, "experts": experts,
                         "network": selection["network"], "model": model,
                         "config_sha256": digest(config, compact=False), "status": "PLANNED"})
    return {"schema_version": 1, "pack": PACK, "study_id": directory.name,
            "created_at": timestamp(), "updated_at": timestamp(), "status": "PLANNED",
            "plan": plan, "plan_sha256": digest(plan), "source_sha256": source_sha256,
            "source_files": files, "children": children}


def validate_ledger(ledger: object, plan: dict, directory: Path) -> dict:
    validate_plan(plan)
    required = {"schema_version", "pack", "study_id", "created_at", "updated_at", "status",
                "plan", "plan_sha256", "source_sha256", "source_files", "children"}
    if not isinstance(ledger, dict) or not required.issubset(ledger) or set(ledger) - required - {"error"}:
        raise ValueError("Invalid study ledger fields")
    if type(ledger["schema_version"]) is not int or ledger["schema_version"] != 1 or ledger["pack"] != PACK or ledger["study_id"] != directory.name or directory.name != plan["study_id"]:
        raise ValueError("Study ledger identity mismatch")
    if ledger["plan"] != plan or ledger["plan_sha256"] != digest(plan):
        raise ValueError("Study plan differs from its frozen plan; start a new study for changed parameters")
    if (not isinstance(ledger["source_files"], list) or not ledger["source_files"]
            or ledger["source_sha256"] != digest(ledger["source_files"])):
        raise ValueError("Study source inventory hash mismatch")
    if ledger["status"] not in {"PLANNED", "RUNNING", "INTERRUPTED", "PASS", "FAIL"}:
        raise ValueError("Invalid study status")
    paths = set()
    for item in ledger["source_files"]:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ValueError("Invalid study source inventory entry")
        if not isinstance(item["path"], str):
            raise ValueError("Invalid study source path")
        path = PurePosixPath(item["path"])
        if (path.is_absolute() or ".." in path.parts or not path.parts
                or str(path) != item["path"] or str(path) in paths
                or type(item["size"]) is not int or item["size"] < 0
                or not isinstance(item["sha256"], str)
                or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])):
            raise ValueError("Invalid or duplicate study source identity")
        paths.add(str(path))
    expected = matrix(plan)
    if not isinstance(ledger["children"], list) or len(ledger["children"]) != len(expected):
        raise ValueError("Study child count differs from the matrix")
    ids = set()
    from scripts.MoE_data_scale.localization_networks.workflow.tasks import resolve_selection
    for child, (seed, fraction, experts) in zip(ledger["children"], expected, strict=True):
        keys = {"run_id", "seed", "train_fraction", "config_sha256", "status", "experts", "network", "model"}
        if not isinstance(child, dict) or not keys.issubset(child) or set(child) - keys - {"attempted_at", "error"}:
            raise ValueError("Invalid study child fields")
        config = child_config(plan, seed, fraction, experts)
        _, selection = selection_for(plan, experts)
        model = resolve_selection(plan["method"], selection["network"])["model"]
        if (type(child["seed"]) is not int or type(child["experts"]) is not int
                or isinstance(child["train_fraction"], bool)
                or child["seed"] != seed or child["train_fraction"] != fraction
                or child["experts"] != experts or child["network"] != selection["network"]
                or child["model"] != model or child["config_sha256"] != digest(config, compact=False)):
            raise ValueError("Study child configuration differs from the frozen matrix")
        validate_run_identity(child["run_id"], PACK, config, selection)
        if child["run_id"] in ids or child["status"] not in CHILD_STATES:
            raise ValueError("Invalid or repeated study child identity/status")
        ids.add(child["run_id"])
        if "attempted_at" in child and (not isinstance(child["attempted_at"], str) or not child["attempted_at"]):
            raise ValueError("Invalid child attempt marker")
        if (child["status"] == "PLANNED") == ("attempted_at" in child):
            raise ValueError("Child attempt marker disagrees with its status; refusing redispatch")
    return ledger


def save(directory: Path, ledger: dict) -> None:
    ledger["updated_at"] = timestamp()
    remote.dump(directory / "study.json", ledger)


def verify_child(ledger: dict, child: dict, config: dict) -> None:
    """Accept a completed child only with the exact frozen source and request."""
    directory, figures = remote.local_paths(child["run_id"], config=config)
    report = read_json(directory / "report.json")
    request = read_json(directory / "request.json")
    manifest = read_json(directory / "manifest.json")
    suite, selection = selection_for(ledger["plan"], child["experts"])
    expected = {"schema_version": 1, "pack": PACK, "run_id": child["run_id"],
                "suite": suite, "config": config, "use_cache": True}
    if selection is not None:
        expected["selection"] = selection
    if request != expected:
        raise ValueError(f"Child request differs from study: {child['run_id']}")
    for value in (report, manifest):
        if (value.get("pack") != PACK or value.get("run_id") != child["run_id"]
                or value.get("suite") != suite or value.get("selection") != selection
                or value.get("source_sha256") != ledger["source_sha256"]
                or value.get("config_sha256") != child["config_sha256"]):
            raise ValueError(f"Child source/configuration identity differs from study: {child['run_id']}")
    if report.get("status") != "PASS" or manifest.get("files") != ledger["source_files"]:
        raise ValueError(f"Child is not a verified PASS for this study: {child['run_id']}")
    verified = remote.verify_visualization_delivery(directory, figures, child["run_id"])
    if verified is None:
        raise ValueError("Study child lacks verified visualization delivery")
