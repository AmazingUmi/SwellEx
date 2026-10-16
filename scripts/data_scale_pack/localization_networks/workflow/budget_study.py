"""Frozen, serial epoch/update matrices with explicit shared observations.

Only fixed architectures are trained. Selection policies are later evaluated
against those observations; they never create extra independent replicates.
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

from scripts.data_scale_pack.localization_networks.workflow import remote, study
from scripts.data_scale_pack.localization_networks.workflow.config import validate_calibration, validate_config
from scripts.data_scale_pack.localization_networks.workflow.tasks import model_plan
from scripts.data_scale_pack.localization_networks.workflow.worker import expected_controlled_budget
from scripts.run_identity import make_run_id, validate_run_identity

PACK = "data_scale_pack"
KIND = "capacity-budget-study-v1"
ARMS = ("epochs", "updates")
OLD_POLICY = {"reference_samples": 600, "exponent": 0.5}
PLAN_FIELDS = {"schema_version", "base_config", "fractions", "seeds", "method",
               "epoch_budget", "update_budget", "validation_interval_updates", "calibration"}
SHARED_REASON = "same_full_fraction_controlled_schedule"


def child_config(plan: dict, seed: int, fraction: float, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown controlled budget arm")
    config = copy.deepcopy(plan["base_config"])
    config["validation"]["seed"] = seed
    config["training"].update(train_fraction=fraction, controlled_batches=True,
                              epochs=plan["epoch_budget"] if arm == "epochs" else None,
                              optimizer_updates=plan["update_budget"] if arm == "updates" else None,
                              validation_interval_updates=plan["validation_interval_updates"] if arm == "updates" else None)
    config["calibration"] = copy.deepcopy(plan["calibration"])
    return validate_config(config)


def validate_plan(plan: object) -> dict:
    if (not isinstance(plan, dict) or set(plan) != PLAN_FIELDS
            or type(plan["schema_version"]) is not int or plan["schema_version"] != 1):
        raise ValueError("Budget study plan requires exactly the documented schema_version=1 fields")
    config = validate_config(plan["base_config"])
    if (config["training"]["train_fraction"] != 1 or config["scaling"] != OLD_POLICY
            or "calibration" in config):
        raise ValueError("Budget base configuration requires train_fraction=1, the original scaling rule, and calibration only at plan level")
    if plan["method"] != "SCM":
        raise ValueError("Capacity budget study supports method=SCM only")
    for key in ("epoch_budget", "update_budget", "validation_interval_updates"):
        if type(plan[key]) is not int or plan[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if (plan["update_budget"] % plan["validation_interval_updates"]
            or plan["update_budget"] // plan["validation_interval_updates"] != plan["epoch_budget"]):
        raise ValueError("Update arm must have the same number of validation checkpoints as epoch_budget")
    fractions, seeds = plan["fractions"], plan["seeds"]
    if (not isinstance(fractions, list) or not fractions
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 < x <= 1 for x in fractions)
            or len(set(fractions)) != len(fractions) or 1.0 not in fractions):
        raise ValueError("Budget fractions must be unique finite values in (0,1] including 1.0")
    if (not isinstance(seeds, list) or not seeds
            or any(type(x) is not int or not 0 <= x <= 2**32-1 for x in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Budget seeds must be unique uint32 integers")
    validate_calibration(plan["calibration"])
    for seed, fraction, arm in matrix(plan):
        child_config(plan, seed, fraction, arm)
    return copy.deepcopy(plan)


def matrix(plan: dict) -> list[tuple[int, float, str]]:
    return [(seed, fraction, arm) for seed in plan["seeds"] for fraction in plan["fractions"]
            for arm in (ARMS if fraction != 1 else ("epochs",))]


def calibration_bytes(plan: dict) -> tuple[bytes, dict]:
    path = remote.PROJECT / plan["calibration"]["path"]
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(remote.PROJECT.resolve()):
        raise ValueError("Calibration artifact is missing or outside the source snapshot")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan["calibration"]["sha256"]:
        raise ValueError("Calibration artifact byte SHA256 differs from plan")
    artifact = study.read_json(path)
    if path.read_bytes() != raw:
        raise ValueError("Calibration artifact changed while being read")
    if not isinstance(artifact, dict) or type(artifact.get("schema_version")) is not int or artifact["schema_version"] != 1:
        raise ValueError("Calibration artifact must have schema_version=1")
    development_seeds = artifact.get("development_seeds")
    if (not isinstance(development_seeds, list) or not development_seeds
            or any(type(seed) is not int or not 0 <= seed <= 2**32-1 for seed in development_seeds)
            or len(set(development_seeds)) != len(development_seeds)):
        raise ValueError("Calibration artifact requires explicit unique development_seeds")
    if set(development_seeds).intersection(plan["seeds"]):
        raise ValueError("Evaluation seeds must be fresh relative to calibration development_seeds")
    policy = artifact.get("policy")
    if not isinstance(policy, dict) or set(policy) != {"reference_samples", "exponent"}:
        raise ValueError("Calibration artifact must expose its two-parameter policy")
    probe = copy.deepcopy(plan["base_config"])
    probe["scaling"] = policy
    validate_config(probe)
    return raw, artifact


def selection_for(plan: dict) -> dict:
    return {"method": plan["method"], "network": None}


def analysis_cells(plan: dict, children: list[dict]) -> list[dict]:
    lookup = {(c["seed"], c["train_fraction"], c["execution_arm"]): c["run_id"] for c in children}
    return [{"seed": seed, "train_fraction": fraction, "arm": arm,
             "run_id": lookup[(seed, fraction, "epochs" if fraction == 1 else arm)],
             "execution_arm": "epochs" if fraction == 1 else arm,
             "shared": fraction == 1, "shared_reason": SHARED_REASON if fraction == 1 else None}
            for seed in plan["seeds"] for fraction in plan["fractions"] for arm in ARMS]


def make_ledger(plan: dict, directory: Path, artifact: dict) -> dict:
    files, source_hash = study.source_inventory()
    matching = [item for item in files if item["path"] == plan["calibration"]["path"]]
    if len(matching) != 1 or matching[0]["sha256"] != plan["calibration"]["sha256"]:
        raise ValueError("Calibration artifact is not frozen in the source inventory")
    children = []
    for seed, fraction, arm in matrix(plan):
        config = child_config(plan, seed, fraction, arm)
        children.append({"run_id": make_run_id(PACK, config, selection_for(plan)), "seed": seed,
                         "train_fraction": fraction, "execution_arm": arm,
                         "config_sha256": study.digest(config, compact=False), "status": "PLANNED"})
    return {"schema_version": 1, "kind": KIND, "pack": PACK, "study_id": directory.name,
            "created_at": study.timestamp(), "updated_at": study.timestamp(), "status": "PLANNED",
            "plan": plan, "plan_sha256": study.digest(plan), "source_sha256": source_hash,
            "source_files": files, "calibration_artifact": artifact, "children": children,
            "analysis_cells": analysis_cells(plan, children)}


def validate_ledger(ledger: object, plan: dict, directory: Path, raw: bytes, artifact: dict) -> dict:
    required = {"schema_version", "kind", "pack", "study_id", "created_at", "updated_at", "status", "plan",
                "plan_sha256", "source_sha256", "source_files", "calibration_artifact", "children", "analysis_cells"}
    if not isinstance(ledger, dict) or not required.issubset(ledger) or set(ledger) - required - {"error"}:
        raise ValueError("Invalid budget study ledger fields")
    if (type(ledger["schema_version"]) is not int or ledger["schema_version"] != 1 or ledger["kind"] != KIND
            or ledger["pack"] != PACK or ledger["study_id"] != directory.name):
        raise ValueError("Budget study identity mismatch")
    if ledger["plan"] != plan or ledger["plan_sha256"] != study.digest(plan):
        raise ValueError("Budget plan differs from frozen plan; changed parameters require a new study")
    saved_artifact = directory / "calibration.json"
    if (saved_artifact.is_symlink() or saved_artifact.read_bytes() != raw
            or ledger["calibration_artifact"] != artifact):
        raise ValueError("Frozen calibration bytes or embedded artifact differ from plan")
    if (not isinstance(ledger["source_files"], list) or not ledger["source_files"]
            or ledger["source_sha256"] != study.digest(ledger["source_files"])):
        raise ValueError("Budget source inventory hash mismatch")
    matching = [item for item in ledger["source_files"] if item["path"] == plan["calibration"]["path"]]
    if len(matching) != 1 or matching[0]["sha256"] != plan["calibration"]["sha256"]:
        raise ValueError("Frozen source does not contain the calibration artifact")
    expected = matrix(plan)
    if not isinstance(ledger["children"], list) or len(ledger["children"]) != len(expected):
        raise ValueError("Budget child count differs from frozen matrix")
    seen = set()
    for child, (seed, fraction, arm) in zip(ledger["children"], expected, strict=True):
        required_child = {"run_id", "seed", "train_fraction", "execution_arm", "config_sha256", "status"}
        if (not isinstance(child, dict) or not required_child.issubset(child)
                or set(child) - required_child - {"attempted_at", "error", "observed_budget", "shared_full_verified"}):
            raise ValueError("Invalid budget child fields")
        config = child_config(plan, seed, fraction, arm)
        if ((child["seed"], child["train_fraction"], child["execution_arm"]) != (seed, fraction, arm)
                or child["config_sha256"] != study.digest(config, compact=False)):
            raise ValueError("Budget child configuration differs from frozen matrix")
        validate_run_identity(child["run_id"], PACK, config, selection_for(plan))
        if child["run_id"] in seen or child["status"] not in study.CHILD_STATES:
            raise ValueError("Invalid or repeated budget child identity/status")
        seen.add(child["run_id"])
        if (child["status"] == "PLANNED") == ("attempted_at" in child):
            raise ValueError("Budget child attempt marker disagrees with status; refusing redispatch")
        if child["status"] == "PASS" and ("observed_budget" not in child or type(child.get("shared_full_verified")) is not bool):
            raise ValueError("PASS budget child lacks verified budget evidence")
    if ledger["analysis_cells"] != analysis_cells(plan, ledger["children"]):
        raise ValueError("Budget analysis cells differ from explicit shared observation mapping")
    return ledger


def verify_child(ledger: dict, child: dict, config: dict) -> None:
    directory, figures = remote.local_paths(child["run_id"], config=config)
    report = study.read_json(directory / "report.json")
    manifest = study.read_json(directory / "manifest.json")
    request = study.read_json(directory / "request.json")
    selection = selection_for(ledger["plan"])
    expected_request = {"schema_version": 1, "pack": PACK, "run_id": child["run_id"], "suite": "budget",
                        "config": config, "selection": selection, "use_cache": True}
    if request != expected_request:
        raise ValueError("Budget child request differs from frozen plan")
    for value in (manifest, report):
        if any(value.get(key) != expected for key, expected in
               (("pack", PACK), ("run_id", child["run_id"]), ("suite", "budget"), ("selection", selection),
                ("source_sha256", ledger["source_sha256"]), ("config_sha256", child["config_sha256"]))):
            raise ValueError("Budget child source/configuration identity differs from study")
    jobs = model_plan("budget", selection)
    if (report.get("status") != "PASS" or manifest.get("files") != ledger["source_files"]
            or report.get("model_plan") != jobs or report.get("calibration") != config["calibration"]):
        raise ValueError("Budget child is not a complete PASS for the frozen plan")
    rows = report.get("metrics")
    if not isinstance(rows, list) or [(row.get("method"), row.get("model")) for row in rows] != [(job["method"], job["model"]) for job in jobs]:
        raise ValueError("Budget child requires exactly the five fixed architecture observations")
    budgets = []
    for row in rows:
        train_count = row.get("scaling", {}).get("train_count")
        if type(train_count) is not int or train_count <= 0:
            raise ValueError("Budget child lacks observed training count")
        budget = expected_controlled_budget(config["training"], train_count)
        if (row.get("budget_plan") != budget or row.get("optimizer_steps") != budget["total_optimizer_updates"]
                or row.get("validation_checks") != budget["validation_checks"]
                or row.get("unique_train_sources_seen") != train_count):
            raise ValueError("Budget child training/validation exposure differs from requested budget")
        budgets.append(budget)
    if any(budget != budgets[0] for budget in budgets):
        raise ValueError("Fixed architectures used different training budgets")
    if any(row["scaling"]["train_count"] != rows[0]["scaling"]["train_count"] for row in rows):
        raise ValueError("Fixed architectures used different retained training counts")
    if child["train_fraction"] == 1:
        plan = ledger["plan"]
        if (budgets[0]["steps_per_data_pass"] != plan["validation_interval_updates"]
                or budgets[0]["total_optimizer_updates"] != plan["update_budget"]
                or budgets[0]["validation_checks"] != plan["epoch_budget"]):
            raise ValueError("Actual full-fraction schedule cannot be shared across the two arms")
    verified = remote.verify_visualization_delivery(directory, figures, child["run_id"])
    if verified is None:
        raise ValueError("Budget child lacks verified visualization delivery")
    child.update(observed_budget=budgets[0], shared_full_verified=child["train_fraction"] == 1)


def execute(directory: Path, ledger: dict) -> int:
    plan = ledger["plan"]
    ledger["status"] = "RUNNING"
    ledger.pop("error", None)
    study.save(directory, ledger)
    for child in ledger["children"]:
        config = child_config(plan, child["seed"], child["train_fraction"], child["execution_arm"])
        if child["status"] == "FAIL":
            ledger.update(status="FAIL", error=f"Child {child['run_id']} failed; no automatic rerun")
            study.save(directory, ledger)
            return 1
        try:
            if child["status"] == "PLANNED":
                _, current = study.source_inventory()
                if current != ledger["source_sha256"]:
                    raise ValueError("Source changed since study creation; refusing to mix source versions")
                if remote.local_paths(child["run_id"], config=config)[0].exists():
                    raise ValueError("Planned budget child already has local evidence; refusing redispatch")
                child.update(status="DISPATCHING", attempted_at=study.timestamp())
                study.save(directory, ledger)
                code = remote.dispatch(config, "budget", False, selection=selection_for(plan), run_id=child["run_id"],
                                       expected_source_sha256=ledger["source_sha256"],
                                       max_connection_failures=study.CONNECTION_FAILURE_LIMIT)
            else:
                report = remote.status(config, child["run_id"])
                child["status"] = report["status"]
                study.save(directory, ledger)
                if report["status"] == "FAIL":
                    remote.fetch(config, child["run_id"])
                    ledger.update(status="FAIL", error=f"Child {child['run_id']} failed; no automatic rerun")
                    study.save(directory, ledger)
                    return 1
                if report["status"] == "PASS":
                    remote.fetch(config, child["run_id"])
                    code = 0
                else:
                    if report["status"] == "STARTING":
                        probe = remote.remote_python(config, study.LAUNCH_PROBE, {"run_dir": remote.run_dir(config, child["run_id"])})
                        if probe.get("launched") is not True:
                            raise RuntimeError("Child has no launch marker; inspect evidence. Study will not relaunch it.")
                    code = remote.follow_run(config, child["run_id"], max_connection_failures=study.CONNECTION_FAILURE_LIMIT)
            if code == 130:
                child["status"], ledger["status"] = "INTERRUPTED", "INTERRUPTED"
                study.save(directory, ledger)
                return 130
            if code:
                report = remote.status(config, child["run_id"])
                child["status"] = "FAIL" if report["status"] == "FAIL" else "UNKNOWN"
                ledger.update(status="FAIL" if child["status"] == "FAIL" else "INTERRUPTED",
                              error=f"Child monitoring exited {code}: {child['run_id']}")
                study.save(directory, ledger)
                return 1
            verify_child(ledger, child, config)
            child["status"] = "PASS"
            child.pop("error", None)
            study.save(directory, ledger)
        except BaseException as exc:
            if "attempted_at" in child and child["status"] != "FAIL":
                child["status"] = "UNKNOWN"
            child["error"] = f"{type(exc).__name__}: {exc}"
            ledger.update(status="FAIL" if child["status"] == "FAIL" else "INTERRUPTED", error=child["error"])
            study.save(directory, ledger)
            if isinstance(exc, KeyboardInterrupt):
                return 130
            raise
    ledger["status"] = "PASS"
    study.save(directory, ledger)
    return 0


def run_budget_study(plan_path: Path, *, study_dir: Path | None = None, dry_run: bool = False) -> int:
    plan = validate_plan(study.read_json(Path(plan_path)))
    raw, artifact = calibration_bytes(plan)
    if study_dir is None:
        study_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-budget-study-" + uuid.uuid4().hex[:8]
        directory = remote.LOCAL_RUNS.resolve() / study_id
    else:
        directory = Path(study_dir).resolve()
        if (Path(study_dir).is_symlink() or directory.parent != remote.LOCAL_RUNS.resolve()
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", directory.name)):
            raise ValueError("--study-dir must name a direct study directory under output/data")
    if dry_run:
        count = len(matrix(plan))
        print(json.dumps({"kind": KIND, "study_dir": str(directory), "plan_sha256": study.digest(plan),
                          "unique_children": count, "unique_model_trainings": 5 * count,
                          "analysis_cells": len(plan["seeds"]) * len(plan["fractions"]) * 2,
                          "analysis_observations": len(plan["seeds"]) * len(plan["fractions"]) * 10,
                          "architectures": model_plan("budget", selection_for(plan)),
                          "shared_fraction": 1.0, "shared_requires_observed_steps_per_data_pass": plan["validation_interval_updates"],
                          "calibration": plan["calibration"], "status": "NOT_RUN"}, indent=2))
        return 0
    is_new = not directory.exists()
    if is_new:
        directory.mkdir(parents=True, exist_ok=False)
    elif not (directory / "study.json").is_file():
        raise ValueError("Existing --study-dir has no study ledger; refusing overwrite")
    print(f"Budget study: {directory}", flush=True)
    with (directory / "study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This budget study already has an active local coordinator") from exc
        if is_new:
            ledger = make_ledger(plan, directory, artifact)
            (directory / "calibration.json").write_bytes(raw)
            study.save(directory, ledger)
        else:
            ledger = validate_ledger(study.read_json(directory / "study.json"), plan, directory, raw, artifact)
        return execute(directory, ledger)
