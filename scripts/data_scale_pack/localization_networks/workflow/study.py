"""Serial, resumable data-size matrices using the isolated haiqin dispatcher.

A child is dispatched at most once. Re-entry follows its existing run ID, and a
failed or ambiguous attempt is never silently replaced with fresh training.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

from scripts.data_scale_pack.localization_networks.workflow import remote
from scripts.data_scale_pack.localization_networks.workflow.config import validate_config
from scripts.run_identity import make_run_id, validate_run_identity

PACK = "data_scale_pack"
PLAN_FIELDS = {"schema_version", "base_config", "fractions", "seeds", "method"}
CHILD_STATES = {"PLANNED", "DISPATCHING", "UNKNOWN", "INTERRUPTED", "STARTING",
                "WAITING_FOR_RESOURCE", "RUNNING", "EXPORTING", "PASS", "FAIL"}
CONNECTION_FAILURE_LIMIT = 3
LAUNCH_PROBE = """import json,pathlib,sys
p=json.load(sys.stdin); run=pathlib.Path(p['run_dir'])
print(json.dumps({'launched':(run/'launch.json').is_file()}))
"""


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
        raise ValueError("Study plan requires exactly schema_version=1, base_config, fractions, seeds, method")
    validate_config(plan["base_config"])
    if plan["base_config"]["training"]["train_fraction"] != 1:
        raise ValueError("Study base_config.training.train_fraction must be 1; fractions controls the matrix")
    if plan["method"] is not None and plan["method"] not in ("RBD", "ELM", "SCM"):
        raise ValueError("Study method must be RBD, ELM, SCM, or null")
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
            child_config(plan, seed, fraction)
    return copy.deepcopy(plan)


def child_config(plan: dict, seed: int, fraction: float) -> dict:
    config = copy.deepcopy(plan["base_config"])
    config["validation"]["seed"] = seed
    config["training"]["train_fraction"] = fraction
    return validate_config(config)


def selection_for(plan: dict) -> tuple[str, dict | None]:
    if plan["method"] is None:
        return "experiment", None
    return "selected", {"method": plan["method"], "network": None}


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
    files, source_sha256 = source_inventory()
    _, selection = selection_for(plan)
    children = []
    for seed in plan["seeds"]:
        for fraction in plan["fractions"]:
            config = child_config(plan, seed, fraction)
            children.append({"run_id": make_run_id(PACK, config, selection), "seed": seed,
                             "train_fraction": fraction, "config_sha256": digest(config, compact=False),
                             "status": "PLANNED"})
    return {"schema_version": 1, "pack": PACK, "study_id": directory.name,
            "created_at": timestamp(), "updated_at": timestamp(), "status": "PLANNED",
            "plan": plan, "plan_sha256": digest(plan), "source_sha256": source_sha256,
            "source_files": files, "children": children}


def validate_ledger(ledger: object, plan: dict, directory: Path) -> dict:
    required = {"schema_version", "pack", "study_id", "created_at", "updated_at", "status",
                "plan", "plan_sha256", "source_sha256", "source_files", "children"}
    if not isinstance(ledger, dict) or not required.issubset(ledger) or set(ledger) - required - {"error"}:
        raise ValueError("Invalid study ledger fields")
    if type(ledger["schema_version"]) is not int or ledger["schema_version"] != 1 or ledger["pack"] != PACK or ledger["study_id"] != directory.name:
        raise ValueError("Study ledger identity mismatch")
    if ledger["plan"] != plan or ledger["plan_sha256"] != digest(plan):
        raise ValueError("Study plan differs from its frozen plan; start a new study for changed parameters")
    if (not isinstance(ledger["source_files"], list) or not ledger["source_files"]
            or ledger["source_sha256"] != digest(ledger["source_files"])):
        raise ValueError("Study source inventory hash mismatch")
    expected = [(seed, fraction) for seed in plan["seeds"] for fraction in plan["fractions"]]
    if not isinstance(ledger["children"], list) or len(ledger["children"]) != len(expected):
        raise ValueError("Study child count differs from the matrix")
    ids = set()
    _, selection = selection_for(plan)
    for child, (seed, fraction) in zip(ledger["children"], expected, strict=True):
        keys = {"run_id", "seed", "train_fraction", "config_sha256", "status"}
        if not isinstance(child, dict) or not keys.issubset(child) or set(child) - keys - {"attempted_at", "error"}:
            raise ValueError("Invalid study child fields")
        config = child_config(plan, seed, fraction)
        if child["seed"] != seed or child["train_fraction"] != fraction or child["config_sha256"] != digest(config, compact=False):
            raise ValueError("Study child configuration differs from the frozen matrix")
        validate_run_identity(child["run_id"], PACK, config, selection)
        if child["run_id"] in ids or child["status"] not in CHILD_STATES:
            raise ValueError("Invalid or repeated study child identity/status")
        ids.add(child["run_id"])
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
    suite, selection = selection_for(ledger["plan"])
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


def execute(directory: Path, ledger: dict) -> int:
    plan = ledger["plan"]
    suite, selection = selection_for(plan)
    ledger["status"] = "RUNNING"
    ledger.pop("error", None)
    save(directory, ledger)
    for child in ledger["children"]:
        config = child_config(plan, child["seed"], child["train_fraction"])
        if child["status"] == "FAIL":
            ledger.update(status="FAIL", error=f"Child {child['run_id']} failed; study does not rerun failed children")
            save(directory, ledger)
            return 1
        try:
            if child["status"] == "PLANNED":
                _, current = source_inventory()
                if current != ledger["source_sha256"]:
                    raise ValueError("Source changed since study creation; refusing to mix source versions")
                if remote.local_paths(child["run_id"], config=config)[0].exists():
                    raise ValueError("Planned child already has local evidence; refusing redispatch")
                child.update(status="DISPATCHING", attempted_at=timestamp())
                save(directory, ledger)
                code = remote.dispatch(config, suite, False, selection=selection, run_id=child["run_id"],
                                       expected_source_sha256=ledger["source_sha256"],
                                       max_connection_failures=CONNECTION_FAILURE_LIMIT)
            else:
                report = remote.status(config, child["run_id"])
                child["status"] = report["status"]
                save(directory, ledger)
                if report["status"] == "FAIL":
                    remote.fetch(config, child["run_id"])
                    ledger.update(status="FAIL", error=f"Child {child['run_id']} failed; no automatic rerun")
                    save(directory, ledger)
                    return 1
                if report["status"] == "PASS":
                    remote.fetch(config, child["run_id"])
                    code = 0
                else:
                    if report["status"] == "STARTING":
                        launch = remote.remote_python(config, LAUNCH_PROBE,
                                                      {"run_dir": remote.run_dir(config, child["run_id"])})
                        if launch.get("launched") is not True:
                            raise RuntimeError("Child initialization has no launch marker; inspect dispatch evidence. Study will not relaunch it.")
                    code = remote.follow_run(config, child["run_id"], max_connection_failures=CONNECTION_FAILURE_LIMIT)
            if code == 130:
                child["status"] = "INTERRUPTED"
                ledger["status"] = "INTERRUPTED"
                save(directory, ledger)
                return 130
            if code:
                report = remote.status(config, child["run_id"])
                child["status"] = "FAIL" if report["status"] == "FAIL" else "UNKNOWN"
                ledger.update(status="FAIL" if child["status"] == "FAIL" else "INTERRUPTED",
                              error=f"Child monitoring exited {code}: {child['run_id']}")
                save(directory, ledger)
                return 1
            verify_child(ledger, child, config)
            child["status"] = "PASS"
            child.pop("error", None)
            save(directory, ledger)
        except BaseException as exc:
            if "attempted_at" in child and child["status"] != "FAIL":
                child["status"] = "UNKNOWN"
            child["error"] = f"{type(exc).__name__}: {exc}"
            ledger.update(status="FAIL" if child["status"] == "FAIL" else "INTERRUPTED", error=child["error"])
            save(directory, ledger)
            if isinstance(exc, KeyboardInterrupt):
                return 130
            raise
    ledger["status"] = "PASS"
    save(directory, ledger)
    return 0


def run_study(plan_path: Path, *, study_dir: Path | None = None) -> int:
    plan = validate_plan(read_json(Path(plan_path)))
    if study_dir is None:
        study_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-scale-study-" + uuid.uuid4().hex[:8]
        directory = remote.LOCAL_RUNS.resolve() / study_id
    else:
        directory = Path(study_dir).resolve()
        if Path(study_dir).is_symlink() or directory.parent != remote.LOCAL_RUNS.resolve() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", directory.name):
            raise ValueError("--study-dir must name a direct study directory under output/data")
    is_new = not directory.exists()
    if is_new:
        directory.mkdir(parents=True, exist_ok=False)
    elif not (directory / "study.json").is_file():
        raise ValueError("Existing --study-dir has no study ledger; refusing to overwrite it")
    print(f"Study: {directory}", flush=True)
    with (directory / "study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This study already has an active local coordinator") from exc
        if is_new:
            ledger = make_ledger(plan, directory)
            save(directory, ledger)
        else:
            ledger = validate_ledger(read_json(directory / "study.json"), plan, directory)
        return execute(directory, ledger)
