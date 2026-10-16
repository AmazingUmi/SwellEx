"""Strict stdlib readers for controlled budget evidence and virtual policy views."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from scripts.data_scale_pack.localization_networks.common.scaling import resolve_capacity, validate_scaling_policy
from scripts.data_scale_pack.localization_networks.workflow import budget_study as contract
from scripts.data_scale_pack.localization_networks.workflow.config import validate_config
from scripts.data_scale_pack.localization_networks.workflow.tasks import model_plan, BUDGET_ARCHITECTURES
from scripts.data_scale_pack.localization_networks.workflow.worker import expected_controlled_budget, validate_controlled_comparison
from scripts.data_scale_pack.visualization import results as run_reader
from scripts.data_scale_pack.visualization.study_results import digest, verify_split, _same_members, _verify_training

PACK = "data_scale_pack"
POLICIES = ("original-policy", "validation-calibrated-v2")
METRICS = ("test_rmse_km", "test_mae_km", "best_val_rmse_km", "parameter_count", "optimizer_steps", "sample_presentations", "total_train_wall_s", "sum_epoch_elapsed_s")
LIMITATIONS = [
    "Fresh seeds reuse the same physical data; this is seed replication, not independent-environment generalization.",
    "Seed variation includes model randomness and training/validation partition variation. Mean ± sample SD is not a confidence interval.",
    "Policies are frozen before this study and reference trained fixed-profile observations; they are not additional independent training runs.",
    "The full-data two-arm comparison shares one execution. Shared observations and identical policy choices are explicitly dependent.",
    "Equal updates and full batches control sample presentations, not unique data coverage or sample reuse frequency.",
    "Epoch and update arms use cosine schedules over their own total update budgets; this compares complete budget protocols.",
    "The calibrated policy was developed with the v1 protocol; evaluation also tests transfer to controlled full batches and per-update scheduling.",
    "The fresh-validation-selected profile is a descriptive reference. Signed test differences to that reference are not test-oracle regret.",
    "Online training metrics are collected while weights update and dropout/BN are in training mode; they are not frozen-model generalization gaps.",
    "Training cost is observed wall time, not a controlled throughput benchmark; shared and virtual views must not be summed as extra compute.",
]


def sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError(f"Invalid {name} SHA256")
    return value


def read_json(path, root, sources):
    return run_reader._json(run_reader._read(Path(path), Path(root).resolve(), sources))


def absolute_sources(root, sources):
    return [{**item, "path": str(Path(root) / item["path"])} for item in sources]


@dataclass
class BudgetRun(run_reader.RunResults):
    observations: list[dict]
    inventories: list[dict]


@dataclass
class BudgetStudy:
    study_path: Path
    study: dict
    unique_rows: list[dict]
    rows: list[dict]
    summaries: list[dict]
    policy_rows: list[dict]
    policy_summaries: list[dict]
    paired_rows: list[dict]
    paired_summaries: list[dict]
    selection: list[dict]
    histories: list[dict]
    sources: list[dict]

    @property
    def status(self):
        return "PASS" if all(row["status"] == "PASS" for row in self.rows) else "PARTIAL"


def verify_controlled(row, model, entry, config, training):
    budget = expected_controlled_budget(config["training"], len(training))
    sidecar = model.architecture
    for key in ("budget_plan", "sampler_plan", "scheduler_plan", "initial_model_state_sha256"):
        if key not in row or sidecar.get(key) != row[key]:
            raise ValueError(f"Controlled sidecar/report mismatch: {key}")
    if row["budget_plan"] != budget or len(model.history) != budget["validation_checks"]:
        raise ValueError("Controlled budget/history differs from configuration")
    seed = config["validation"]["seed"]
    sampler = {"version": "independent-torch-generators-v1", "train_seed": seed+1000003,
               "validation_seed": seed+2000003, "batch_digest": "sha256-json-source-id-batches-v1"}
    if row["sampler_plan"] != sampler:
        raise ValueError("Controlled sampler differs from frozen independent generators")
    scheduler = row["scheduler_plan"]
    initial_lr = run_reader.finite(scheduler.get("initial_lr"), "initial learning rate", nonnegative=True)
    total, interval, batch = budget["total_optimizer_updates"], budget["validation_interval_updates"], budget["batch_size"]
    if initial_lr <= 0 or scheduler != {"policy": "cosine_per_update_v1", "step_unit": "optimizer_update", "total_updates": total, "initial_lr": initial_lr, "eta_min": 0.0}:
        raise ValueError("Controlled scheduler differs from per-update cosine contract")
    sha(row["initial_model_state_sha256"], "initial model state")
    seen, next_lr, lr_digest = 0, initial_lr, hashlib.sha256()
    for index, record in enumerate(model.history, 1):
        updates = index*interval
        required = {"epoch": index, "record_unit": "validation_checkpoint", "validation_index": index,
                    "optimizer_steps_epoch": interval, "optimizer_steps": updates,
                    "sample_presentations_record": interval*batch, "sample_presentations": updates*batch,
                    "data_passes_completed": updates//budget["steps_per_data_pass"],
                    "updates_in_current_pass": updates % budget["steps_per_data_pass"]}
        if any(type(record.get(key)) is not type(value) or record[key] != value for key, value in required.items()):
            raise ValueError("Controlled history counters or record unit are inconsistent")
        new_seen = record.get("unique_train_sources_seen")
        if type(new_seen) is not int or not max(seen, batch) <= new_seen <= min(len(training), updates*batch):
            raise ValueError("Controlled history unique-source count is invalid")
        seen = new_seen
        first_lr = next_lr
        for update in range(updates-interval+1, updates+1):
            last_lr = next_lr
            lr_digest.update((json.dumps(last_lr, allow_nan=False)+"\n").encode())
            next_lr = ((1+math.cos(math.pi*update/total))/(1+math.cos(math.pi*(update-1)/total)))*last_lr
        for key, expected in (("lr", first_lr), ("last_update_lr", last_lr), ("next_lr", next_lr)):
            if not math.isclose(run_reader.finite(record.get(key), key, nonnegative=True), expected, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("Controlled learning-rate values differ from cosine trajectory")
        if record.get("lr_sequence_sha256") != lr_digest.hexdigest():
            raise ValueError("Controlled learning-rate sequence hash differs")
        for key in ("batch_sequence_sha256", "record_batch_sequence_sha256", "lr_sequence_sha256", "seen_source_identity_sha256"):
            sha(record.get(key), key)
    expected_seen = hashlib.sha256(json.dumps(sorted(training), separators=(",", ":")).encode()).hexdigest()
    last, best = model.history[-1], min(model.history, key=lambda item: item["val_rmse_km"])
    pass_hashes = row.get("data_pass_batch_sequence_sha256")
    if not isinstance(pass_hashes, list) or len(pass_hashes) != total // budget["steps_per_data_pass"]:
        raise ValueError("Controlled complete-pass batch hash list has invalid length")
    for value in pass_hashes:
        sha(value, "complete-pass batch sequence")
    for record in model.history:
        if record["updates_in_current_pass"] == 0 and pass_hashes[record["data_passes_completed"]-1] != record["batch_sequence_sha256"]:
            raise ValueError("Complete-pass batch hashes differ from validation history boundaries")
    expected = {"history_record_unit": "validation_checkpoint", "validation_checks": len(model.history),
                "best_validation_index": best["validation_index"], "best_optimizer_step": best["optimizer_steps"],
                "optimizer_steps": total, "sample_presentations": total*batch, "unique_train_sources_seen": len(training),
                "final_batch_sequence_sha256": last["batch_sequence_sha256"], "final_lr_sequence_sha256": last["lr_sequence_sha256"],
                "final_seen_source_identity_sha256": expected_seen}
    if seen != len(training) or last["seen_source_identity_sha256"] != expected_seen or any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("Controlled final exposure or best-checkpoint summary differs")
    metrics = _verify_training(row, model, entry)
    return {**metrics, **expected, "budget_plan": budget, "sampler_plan": sampler, "scheduler_plan": scheduler,
            "initial_model_state_sha256": row["initial_model_state_sha256"], "data_pass_batch_sequence_sha256": pass_hashes}


def load_budget_run(run_dir: Path, *, require_pass=True) -> BudgetRun:
    root, sources = Path(run_dir).resolve(), []
    raw = run_reader._read(root/"report.json", root, sources)
    report = run_reader._json(raw)
    allowed = {"PASS"} if require_pass else {"PASS", "RUNNING"}
    if (not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("pack") != PACK
            or report.get("run_id") != root.name or report.get("suite") != "budget" or report.get("status") not in allowed
            or report.get("purpose") != "controlled_epoch_vs_update_budget_with_frozen_validation_calibration"):
        raise ValueError("Budget rendering requires its canonical complete run identity/status")
    jobs, config = run_reader.recorded_contract(report, root, sources)
    validate_config(copy.deepcopy(config))
    if not config["training"].get("controlled_batches") or report.get("calibration") != config.get("calibration"):
        raise ValueError("Budget run requires controlled batches and matching calibration provenance")
    rows = report.get("metrics")
    if not isinstance(rows, list) or [(r.get("method"), r.get("model")) for r in rows] != [(j["method"], j["model"]) for j in jobs]:
        raise ValueError("Budget run must contain its complete ordered five-profile plan")
    checks = report.get("checks", [])
    if not isinstance(checks, list) or any(not isinstance(c, dict) or c.get("status") == "FAIL" for c in checks):
        raise ValueError("Budget run contains failed checks")
    completed = {c.get("name") for c in checks if c.get("status") == "PASS"}
    archive = read_json(root/"reports/training_inventory.json", root, sources)
    if (archive.get("schema_version") != 1 or archive.get("pack") != PACK or archive.get("run_id") != root.name
            or not isinstance(archive.get("models"), list)):
        raise ValueError("Budget training inventory identity is invalid")
    entries = archive["models"]
    if [(e.get("method"), e.get("model")) for e in entries] != [(j["method"], j["model"]) for j in jobs]:
        raise ValueError("Budget inventory differs from complete model plan")
    models, observations = [], []
    baseline_members = baseline_sources = None
    for job, row, entry in zip(jobs, rows, entries, strict=True):
        method, model = job["method"], job["model"]
        if f"evaluate_{method}_{model}" not in completed:
            raise ValueError("Budget model evaluation has not passed")
        directory = root/"project/outputs/networks_results"/method/model
        path = run_reader._unique(directory/"test_outputs", "*_predictions_*.csv")
        predictions = run_reader.read_predictions(run_reader._read(path, root, sources))
        metrics = run_reader.summarize(predictions)
        run_reader._verify_metrics(metrics, row)
        path = run_reader._unique(directory/"train_outputs", "*_history_*.json")
        history = run_reader.read_history(run_reader._read(path, root, sources), row.get("validation_checks"))
        path = run_reader._unique(directory/"train_outputs", "*_architecture_*.json")
        if row.get("architecture_path") != path.relative_to(root).as_posix():
            raise ValueError("Budget architecture path differs from canonical model output")
        architecture = run_reader.read_architecture(run_reader._read(path, root, sources), row, job, config)
        result = run_reader.ModelResult(method, model, predictions, history, metrics, architecture)
        inventory = entry.get("training_inventory")
        if (inventory != architecture.get("training_inventory") or not isinstance(inventory, dict) or inventory.get("schema_version") != 1
                or entry.get("seed") != config["validation"]["seed"] or entry.get("train_fraction") != config["training"]["train_fraction"]
                or row.get("training_inventory_path") != "reports/training_inventory.json" or entry.get("sampling") != architecture["sampling"]):
            raise ValueError("Budget physical inventory disagrees with sidecar/configuration")
        training, validation, testing = (verify_split(value, label) for value, label in
                                       ((inventory["train"], "training"), (inventory["validation"], "validation"), (entry["test"], "test")))
        if set(training)&(set(validation)|set(testing)) or set(validation)&set(testing):
            raise ValueError("Budget physical splits overlap")
        if baseline_members is None:
            baseline_members = (training, validation, testing)
        for actual, expected in zip((training, validation, testing), baseline_members, strict=True):
            _same_members(actual, expected, "Same-run controlled model")
        sampling, scaling = architecture["sampling"], architecture["scaling"]
        if (len(training) != scaling["train_count"] or len(training) != math.floor(sampling["candidate_train_count"]*sampling["train_fraction"])
                or len(validation) != sampling["validation_count"] or row.get("parameter_count") != scaling["parameter_count"]):
            raise ValueError("Budget actual training count or parameter count differs")
        for key, members in (("train_indices", training), ("val_indices", validation)):
            indices = entry.get(key)
            if not isinstance(indices, list) or len(indices) != len(members) or len(set(indices)) != len(indices) or any(type(x) is not int or x < 0 for x in indices):
                raise ValueError("Budget physical source indices are invalid")
        for key, split in (("train_identity_sha256", inventory["train"]), ("val_identity_sha256", inventory["validation"]), ("test_identity_sha256", entry["test"])):
            if row.get(key) != split["identity_sha256"]:
                raise ValueError("Budget row physical identity differs from inventory")
        if len(predictions) != len(testing) or any(p.source_index not in testing or abs(p.time_s-testing[p.source_index][0]) > 1.1e-6 or abs(p.true_km-testing[p.source_index][1]) > 1.1e-6 for p in predictions):
            raise ValueError("Budget test CSV differs from full-precision inventory")
        for members, coverage in ((validation, inventory.get("validation_coverage")), (testing, entry.get("test_coverage"))):
            low, high = min(v[1] for v in training.values()), max(v[1] for v in training.values())
            within = sum(low <= v[1] <= high for v in members.values())
            if coverage != {"within_train_range_count": within, "within_train_range_fraction": within/len(members)}:
                raise ValueError("Budget held-out range coverage differs")
        normalized = normalized_sources(entry)
        if baseline_sources is not None and normalized != baseline_sources:
            raise ValueError("Fixed models used different source dataset bytes")
        baseline_sources = normalized
        evidence = verify_controlled(row, result, entry, config, training)
        common = {"run_id": root.name, "observation_id": f"{root.name}/{method}/{model}", **job,
                  "seed": config["validation"]["seed"], "train_fraction": sampling["train_fraction"],
                  "execution_arm": evidence["budget_plan"]["mode"], "status": "PASS", "reason": None,
                  "train_count": len(training), "validation_count": len(validation), "test_count": len(testing),
                  "selected_profile": scaling["selected_profile"], "parameter_count": scaling["parameter_count"],
                  "blocks": sum(scaling["stage_repeats"]), "test_rmse_km": metrics["rmse_km"], "test_mae_km": metrics["mae_km"],
                  "base_channels": config["training"]["base_channels"], **evidence,
                  **{key: row[key] for key in ("train_identity_sha256", "val_identity_sha256", "test_identity_sha256")}}
        observations.append(common)
        models.append(result)
    validate_controlled_comparison(rows)
    keys = ("batch_sequence_sha256", "record_batch_sequence_sha256", "seen_source_identity_sha256", "lr_sequence_sha256")
    for model in models[1:]:
        if any(any(a[key] != b[key] for key in keys) for a, b in zip(models[0].history, model.history, strict=True)):
            raise ValueError("Fixed models have different per-validation controlled streams")
    return BudgetRun(root, report, raw, models, sources, config, observations, entries)


def normalized_sources(entry):
    identity = entry.get("source_identity")
    if not isinstance(identity, dict) or identity.get("version") != 1:
        raise ValueError("Budget training source byte identity is missing")
    result = {}
    for key in ("train_sources", "validation_sources"):
        items = identity.get(key)
        if not isinstance(items, list) or not items:
            raise ValueError("Budget training source inventory is missing")
        for item in items:
            if type(item.get("size_bytes")) is not int or item["size_bytes"] <= 0:
                raise ValueError("Invalid training-source byte size")
            sha(item.get("sha256"), "training source")
        result[key] = [{k: item[k] for k in ("size_bytes", "sha256")} for item in items]
    return result


def summaries(rows, keys, seeds, metrics=METRICS):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for values, group in grouped.items():
        if len({row["seed"] for row in group}) != len(group):
            raise ValueError("Duplicate seed observation in an aggregation group")
        passed = [row for row in group if row["status"] == "PASS"]
        counts = {row["train_count"] for row in passed if "train_count" in row}
        if len(counts) > 1:
            raise ValueError("Actual training counts differ across seeds")
        item = {**dict(zip(keys, values)), "status": "PASS" if len(passed) == len(seeds) else "PARTIAL" if passed else "NOT_RUN",
                "n_seeds": len(passed), "planned_seeds": len(seeds), "observed_seeds": [row["seed"] for row in passed],
                "missing_seeds": [seed for seed in seeds if seed not in {row["seed"] for row in passed}],
                "train_count": next(iter(counts)) if counts else None,
                "observation_ids": [row.get("observation_id") for row in passed],
                "profiles": sorted({row["selected_profile"] for row in passed if "selected_profile" in row})}
        for metric in metrics:
            vals = [row[metric] for row in passed]
            item[metric+"_mean"] = statistics.mean(vals) if vals else None
            item[metric+"_sd"] = statistics.stdev(vals) if len(vals) > 1 else None
        output.append(item)
    return output


def policy_views(rows, plan, artifact):
    seeds, fractions = plan["seeds"], plan["fractions"]
    lookup = {(r["arm"], r["train_fraction"], r["seed"], r["architecture"]): r for r in rows}
    groups = summaries(rows, ("arm", "train_fraction", "architecture"), seeds)
    summary_map = {(r["arm"], r["train_fraction"], r["architecture"]): r for r in groups}
    policy_rows, selection, paired = [], [], []
    policies = {POLICIES[0]: contract.OLD_POLICY, POLICIES[1]: artifact["policy"]}
    for arm in contract.ARMS:
        for fraction in fractions:
            candidates = [summary_map[(arm, fraction, arch)] for arch in BUDGET_ARCHITECTURES if arch != "complex_CNN_baseline"]
            winner = min(candidates, key=lambda r: (r["best_val_rmse_km_mean"], r["parameter_count_mean"])) if all(r["status"] == "PASS" for r in candidates) else None
            selection.append({"arm": arm, "train_fraction": fraction, "status": "PASS" if winner else "NOT_RUN",
                              "selected_architecture": winner["architecture"] if winner else None,
                              "criterion": "mean fresh best-validation RMSE, exact ties fewer parameters; descriptive",
                              **({"train_count": winner["train_count"], "selected_profile": winner["architecture"].removeprefix("complex_CNN_"),
                                  **{key: winner[key] for metric in METRICS for key in (metric+"_mean", metric+"_sd")}} if winner else {})})
            for seed in seeds:
                mapped = {}
                for policy_id, policy in policies.items():
                    fixed = [lookup[(arm, fraction, seed, arch)] for arch in BUDGET_ARCHITECTURES]
                    common = {"arm": arm, "train_fraction": fraction, "seed": seed, "policy_id": policy_id,
                              "virtual": True, "independent_training": False, "policy": policy}
                    if not all(r["status"] == "PASS" for r in fixed):
                        item = {**common, "status": "NOT_RUN", "reason": "incomplete_fixed_profile_cell"}
                    else:
                        probe = fixed[0]
                        profile = resolve_capacity("auto", train_count=probe["train_count"], base_channels=probe["base_channels"], scaling_policy=policy)["selected_profile"]
                        source = lookup[(arm, fraction, seed, "complex_CNN_"+profile)]
                        item = {**source, **common, "source_observation_id": source["observation_id"]}
                    policy_rows.append(item)
                    mapped[policy_id] = item
                    reference = lookup[(arm, fraction, seed, winner["architecture"])] if winner else {"status": "NOT_RUN"}
                    paired.append(pair(item, reference, "policy-minus-fresh-validation-reference", policy_id))
                left, right = mapped[POLICIES[1]], mapped[POLICIES[0]]
                paired.append(pair(left, right, "calibrated-minus-original", "policy-comparison"))
    for fraction in fractions:
        for seed in seeds:
            for arch in BUDGET_ARCHITECTURES:
                paired.append(pair(lookup[("updates", fraction, seed, arch)], lookup[("epochs", fraction, seed, arch)], "updates-minus-epochs", arch))
    return groups, policy_rows, summaries(policy_rows, ("arm", "train_fraction", "policy_id"), seeds), selection, paired


def pair(left, right, comparison, series):
    result = {"comparison": comparison, "series": series, "seed": left["seed"], "arm": left["arm"],
              "train_fraction": left["train_fraction"], "status": "NOT_RUN", "same_observation": None,
              "left_observation_id": left.get("observation_id"), "right_observation_id": right.get("observation_id")}
    if left["status"] == right["status"] == "PASS":
        same = left["observation_id"] == right["observation_id"]
        result.update(status="PASS", train_count=left["train_count"], same_observation=same,
                      **{metric+"_difference": left[metric]-right[metric] for metric in METRICS})
        if same and any(result[m+"_difference"] != 0 for m in METRICS):
            raise ValueError("A shared physical observation has inconsistent virtual metrics")
    return result


def load_budget_study(study_path: Path, runs_root: Path | None = None) -> BudgetStudy:
    path = Path(study_path).resolve()
    sources = []
    ledger = read_json(path, path.parent, sources)
    plan = contract.validate_plan(ledger.get("plan"))
    raw = run_reader._read(path.parent/"calibration.json", path.parent, sources)
    artifact = run_reader._json(raw)
    if hashlib.sha256(raw).hexdigest() != plan["calibration"]["sha256"]:
        raise ValueError("Calibration bytes differ from frozen SHA256")
    contract.validate_ledger(ledger, plan, path.parent, raw, artifact)
    if artifact.get("policy_id") != POLICIES[1] or artifact.get("schema_version") != 1:
        raise ValueError("Unsupported calibration artifact identity")
    validate_scaling_policy(artifact.get("policy"))
    development = artifact.get("development_seeds")
    if not isinstance(development, list) or not development or any(type(s) is not int or not 0 <= s <= 2**32-1 for s in development) or len(set(development)) != len(development) or set(development)&set(plan["seeds"]):
        raise ValueError("Evaluation seeds are not fresh relative to calibration")
    sources = absolute_sources(path.parent, sources)
    root = Path(runs_root).resolve() if runs_root else path.parent / "children"
    jobs = model_plan("budget", contract.selection_for(plan))
    unique, histories, inventory_map, run_results = [], [], {}, {}
    frozen_identity = lock = normalized = None
    for child in ledger["children"]:
        base = {"run_id": child["run_id"], "seed": child["seed"], "train_fraction": child["train_fraction"], "execution_arm": child["execution_arm"], "run_status": child["status"]}
        if child["status"] != "PASS":
            unique.extend({**base, **job, "observation_id": f"{child['run_id']}/{job['method']}/{job['model']}", "status": "NOT_RUN", "reason": child.get("error", "child_not_pass")} for job in jobs)
            continue
        directory = root/child["run_id"]
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("PASS budget child evidence is missing or external")
        run = load_budget_run(directory)
        expected = contract.child_config(plan, child["seed"], child["train_fraction"], child["execution_arm"])
        if run.config != expected or run.report["source_sha256"] != ledger["source_sha256"] or run.report.get("calibration") != plan["calibration"]:
            raise ValueError("Budget child config/source/calibration differs from study")
        manifest = read_json(directory/"manifest.json", directory, run.sources)
        request = read_json(directory/"request.json", directory, run.sources)
        if manifest.get("files") != ledger["source_files"] or request != {"schema_version": 1, "pack": PACK, "run_id": child["run_id"], "suite": "budget", "selection": contract.selection_for(plan), "config": expected, "use_cache": True}:
            raise ValueError("Budget child request/source inventory differs from exact plan")
        observed_lock = sha(run.report.get("uv_lock_sha256"), "lockfile")
        if lock is not None and lock != observed_lock:
            raise ValueError("Budget children used different environment locks")
        lock = observed_lock
        identity = read_json(directory/"data_identity.json", directory, run.sources)
        if identity.get("schema_version") != 1 or identity.get("pack") != PACK or digest({k:v for k,v in identity.items() if k != "cache_key"}) != identity.get("cache_key") or run.report.get("dataset_cache", {}).get("cache_key") != identity["cache_key"]:
            raise ValueError("Budget raw-input/cache identity is invalid")
        if frozen_identity is not None and identity != frozen_identity:
            raise ValueError("Budget children used different raw inputs/generation")
        frozen_identity = identity
        budget = run.observations[0]["budget_plan"]
        if child.get("observed_budget") != budget or child.get("shared_full_verified") is not (child["train_fraction"] == 1):
            raise ValueError("Ledger observed budget or shared verification differs from evidence")
        if child["train_fraction"] == 1 and (budget["steps_per_data_pass"] != plan["validation_interval_updates"] or budget["total_optimizer_updates"] != plan["update_budget"] or budget["validation_checks"] != plan["epoch_budget"]):
            raise ValueError("Full-data arms are not the same controlled schedule")
        for observation, model, entry in zip(run.observations, run.models, run.inventories, strict=True):
            population = normalized_sources(entry)
            if normalized is not None and population != normalized:
                raise ValueError("Budget source dataset byte identities differ")
            normalized = population
            unique.append({**base, **observation})
            inventory_map[observation["observation_id"]] = entry
            histories.extend({**base, **{k:observation[k] for k in ("observation_id", "architecture", "model", "method", "train_count")}, **record} for record in model.history)
        sources.extend(absolute_sources(directory, run.sources))
        run_results[child["run_id"]] = run
    by_observation = {r["observation_id"]: r for r in unique}
    logical = []
    for cell in ledger["analysis_cells"]:
        for job in jobs:
            source = by_observation[f"{cell['run_id']}/{job['method']}/{job['model']}"]
            logical.append({**source, **cell, "analysis_cell_id": f"{cell['seed']}/{cell['train_fraction']:g}/{cell['arm']}/{job['architecture']}"})
    train_members, val_members, test_members = {}, {}, None
    arm_streams = {}
    for observation in unique:
        if observation["status"] != "PASS":
            continue
        entry = inventory_map[observation["observation_id"]]
        train = verify_split(entry["training_inventory"]["train"], "training")
        val = verify_split(entry["training_inventory"]["validation"], "validation")
        test = verify_split(entry["test"], "test")
        seed, fraction = observation["seed"], observation["train_fraction"]
        for mapping, key, members, label in ((train_members, (seed, fraction), train, "Same-cell cross-arm training"), (val_members, seed, val, "Same-seed validation")):
            if key in mapping:
                _same_members(mapping[key], members, label)
            mapping[key] = members
        if test_members is not None:
            _same_members(test_members, test, "Cross-seed/arm test")
        test_members = test
        key = (seed, fraction, observation["architecture"])
        records = {r["optimizer_steps"]:r for r in histories if r["observation_id"] == observation["observation_id"]}
        if key in arm_streams:
            prior, prior_records = arm_streams[key]
            if observation["initial_model_state_sha256"] != prior["initial_model_state_sha256"]:
                raise ValueError("Paired budget arms have different initial model states")
            for step in set(records)&set(prior_records):
                if records[step]["batch_sequence_sha256"] != prior_records[step]["batch_sequence_sha256"]:
                    raise ValueError("Paired budget arms have different batch-stream prefixes")
            left, right = observation["data_pass_batch_sequence_sha256"], prior["data_pass_batch_sequence_sha256"]
            count = min(len(left), len(right))
            if count == 0 or left[:count] != right[:count]:
                raise ValueError("Paired budget arms have different complete-pass batch-stream prefixes")
        arm_streams[key] = observation, records
    for seed in plan["seeds"]:
        available = [(f, train_members[(seed,f)]) for f in sorted(plan["fractions"]) if (seed,f) in train_members]
        for (_, small), (_, large) in zip(available, available[1:]):
            if not set(small).issubset(large) or any(large[k] != v for k,v in small.items()):
                raise ValueError("Budget training fractions are not physically nested")
    grouped, virtual, policy_grouped, selection, paired = policy_views(logical, plan, artifact)
    paired_grouped = summaries(paired, ("comparison", "series", "arm", "train_fraction"), plan["seeds"], tuple(m+"_difference" for m in METRICS))
    return BudgetStudy(path, ledger, unique, logical, grouped, virtual, policy_grouped, paired, paired_grouped, selection, histories, sources)
