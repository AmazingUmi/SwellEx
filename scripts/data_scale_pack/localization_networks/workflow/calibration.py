"""Calibrate a capacity rule from frozen development-validation evidence only.

The fit API accepts an exact projection schema, never study rows containing
held-out metrics. This module and its command-line interface use only stdlib.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from scripts.data_scale_pack.localization_networks.common import scaling
from scripts.data_scale_pack.localization_networks.workflow import study
from scripts.run_identity import validate_run_identity

DEVELOPMENT_SEEDS = (2026, 2027, 2028)
COUNTS = (337, 675, 1350, 2700)
FRACTIONS = (.125, .25, .5, 1.)
PROFILES = ("small", "medium", "large", "xlarge")
REFERENCE_GRID = tuple(range(600, 1801, 100))
EXPONENT_GRID = tuple(value / 100 for value in range(25, 151, 5))
PROJECTION_FIELDS = {"seed", "train_count", "profile", "validation_rmse_km", "parameter_count", "run_id", "train_fraction"}
TOLERANCE = 1e-12
BASE_CHANNELS = 16
OLD_POLICY = {"reference_samples": 600, "exponent": .5}
ARTIFACT_NAME = "scm_capacity_calibration_v2.json"
TIE_RULE = {
    "score_absolute_tolerance": TOLERANCE,
    "order": ["retain scores within tolerance of global minimum", "lowest mean parameter count over N",
              "within each identical N-to-profile mapping: largest minimum absolute log distance to a tier threshold",
              "lowest exponent", "lowest reference_samples"],
    "threshold_profiles": ["medium", "large", "xlarge"],
    "threshold_formula": "reference_samples * (P_profile / P_medium) ** (1 / exponent)",
    "margin_formula": "min over observed N and three tier thresholds of abs(log(N / threshold))",
}


def _read(path: Path) -> tuple[object, dict]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or symlink calibration input: {path}")
    raw = path.read_bytes()
    value = study.read_json(path)
    if path.read_bytes() != raw:
        raise ValueError(f"Calibration input changed while being read: {path}")
    return value, {"path": str(path.resolve()), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _positive_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def validate_projection(rows: object, *, seeds=DEVELOPMENT_SEEDS) -> list[dict]:
    """Reject broader rows so selection cannot accidentally consume test scores."""
    if (not isinstance(seeds, (list, tuple)) or not seeds or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds) or not set(seeds).issubset(DEVELOPMENT_SEEDS)):
        raise ValueError("Projection seeds must be a nonempty subset of development seeds")
    if not isinstance(rows, list) or len(rows) != len(seeds) * len(COUNTS) * len(PROFILES):
        raise ValueError("Development projection must contain the complete seed/count/fixed-profile grid")
    expected = {(seed, count, profile) for seed in seeds for count in COUNTS for profile in PROFILES}
    seen, run_ids = set(), {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != PROJECTION_FIELDS:
            raise ValueError("Fit accepts only the exact validation-only projection fields")
        if type(row["seed"]) is not int or type(row["train_count"]) is not int or not isinstance(row["profile"], str):
            raise ValueError("Invalid projection identity types")
        key = row["seed"], row["train_count"], row["profile"]
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate development projection cell")
        seen.add(key)
        fraction = FRACTIONS[COUNTS.index(row["train_count"])]
        if type(row["train_fraction"]) not in (int, float) or row["train_fraction"] != fraction:
            raise ValueError("Projection fraction differs from its actual training count")
        _positive_number(row["validation_rmse_km"], "validation_rmse_km")
        if type(row["parameter_count"]) is not int or row["parameter_count"] != scaling.parameter_count(row["profile"], BASE_CHANNELS):
            raise ValueError("Projection parameter count differs from the base_channels=16 fixed architecture")
        if not isinstance(row["run_id"], str) or not row["run_id"]:
            raise ValueError("Projection needs a source run identity")
        cell = row["seed"], row["train_count"]
        if cell in run_ids and run_ids[cell] != row["run_id"]:
            raise ValueError("Profiles in one development cell have different source runs")
        run_ids[cell] = row["run_id"]
    if seen != expected or len(set(run_ids.values())) != len(run_ids):
        raise ValueError("Development projection grid or source run identities are incomplete")
    return sorted(copy.deepcopy(rows), key=lambda row: (row["seed"], row["train_count"], PROFILES.index(row["profile"])))


def load_development_projection(analysis_path: Path, *, study_path: Path | None = None) -> tuple[list[dict], dict]:
    analysis_path = Path(analysis_path)
    study_path = Path(study_path) if study_path is not None else analysis_path.parent.parent / "study.json"
    analysis, analysis_source = _read(analysis_path)
    manifest, manifest_source = _read(analysis_path.parent / "manifest.json")
    ledger, study_source = _read(study_path)
    for value in (analysis, manifest, ledger):
        if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
                or value["schema_version"] != 1 or value.get("pack") != "data_scale_pack" or value.get("status") != "PASS"):
            raise ValueError("Calibration requires a complete PASS source study and PASS analysis")
    if analysis.get("study") != ledger or any(value.get("study_id") != ledger.get("study_id") for value in (analysis, manifest)):
        raise ValueError("Analysis embedded study differs from its source ledger")
    plan = study.validate_plan(ledger["plan"])
    config = plan["base_config"]
    if (plan["method"] != "SCM" or plan["seeds"] != list(DEVELOPMENT_SEEDS) or plan["fractions"] != list(FRACTIONS)
            or config["training"]["base_channels"] != BASE_CHANNELS or config["training"].get("controlled_batches", False)):
        raise ValueError("Calibration requires the original SCM, three-seed, four-fraction, base-16 development study")
    source_files = ledger.get("source_files")
    if (not isinstance(source_files, list) or not source_files or ledger.get("source_sha256") != study.digest(source_files)
            or ledger.get("plan_sha256") != study.digest(plan)
            or manifest.get("source_sha256") != ledger["source_sha256"] or manifest.get("plan_sha256") != ledger["plan_sha256"]):
        raise ValueError("Analysis/source study hashes are inconsistent")
    inventory_paths = [item.get("path") for item in source_files if isinstance(item, dict)]
    if len(inventory_paths) != len(source_files) or len(set(inventory_paths)) != len(inventory_paths):
        raise ValueError("Development source inventory contains duplicate or invalid entries")
    outputs = manifest.get("files")
    matches = [item for item in outputs if isinstance(item, dict) and item.get("path") == analysis_path.name] if isinstance(outputs, list) else []
    if len(matches) != 1 or any(matches[0].get(key) != analysis_source[key] for key in ("size", "sha256")):
        raise ValueError("Analysis file bytes differ from their rendering manifest")
    inputs = manifest.get("inputs")
    matches = [item for item in inputs if isinstance(item, dict) and isinstance(item.get("path"), str)
               and Path(item["path"]).name == "study.json"] if isinstance(inputs, list) else []
    if len(matches) != 1 or any(matches[0].get(key) != study_source[key] for key in ("size", "sha256")):
        raise ValueError("Rendering manifest source study hash differs from the supplied ledger")
    children = ledger.get("children")
    if not isinstance(children, list) or len(children) != len(DEVELOPMENT_SEEDS) * len(FRACTIONS):
        raise ValueError("Development study requires all twelve source children")
    lookup, run_ids = {}, set()
    selection = {"method": "SCM", "network": None}
    for child in children:
        if not isinstance(child, dict) or child.get("status") != "PASS":
            raise ValueError("Every calibration development child must be PASS")
        key = child.get("seed"), child.get("train_fraction")
        if key[0] not in DEVELOPMENT_SEEDS or key[1] not in FRACTIONS or key in lookup or child.get("run_id") in run_ids:
            raise ValueError("Duplicate or unexpected calibration source child")
        child_config = study.child_config(plan, *key)
        validate_run_identity(child["run_id"], "data_scale_pack", child_config, selection)
        if child.get("config_sha256") != study.digest(child_config, compact=False):
            raise ValueError("Development child configuration hash differs from source plan")
        lookup[key] = child
        run_ids.add(child["run_id"])
    metrics = analysis.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError("Analysis has no development observations")
    projection = []
    for row in metrics:
        if not isinstance(row, dict):
            raise ValueError("Invalid development observation")
        architecture = row.get("architecture")
        if architecture not in {"complex_CNN_" + profile for profile in PROFILES}:
            continue
        profile = architecture.removeprefix("complex_CNN_")
        child = lookup.get((row.get("seed"), row.get("train_fraction")))
        if (child is None or row.get("run_id") != child["run_id"] or row.get("study_id") != ledger["study_id"]
                or row.get("status") != "PASS" or row.get("run_status") != "PASS" or row.get("method") != "SCM"
                or row.get("selection_mode") != "fixed" or row.get("selected_profile") != profile
                or row.get("model") != "scm_complex_cnn_" + profile + "_range"):
            raise ValueError("Fixed-profile observation differs from its source child or architecture")
        projection.append({"seed": row["seed"], "train_count": row.get("train_count"), "profile": profile,
                           "validation_rmse_km": row.get("best_val_rmse_km"), "parameter_count": row.get("parameter_count"),
                           "run_id": row["run_id"], "train_fraction": row["train_fraction"]})
    projection = validate_projection(projection)
    return projection, {"analysis": analysis_source, "analysis_manifest": manifest_source, "study": study_source,
                        "study_id": ledger["study_id"], "training_source_sha256": ledger["source_sha256"],
                        "training_plan_sha256": ledger["plan_sha256"]}


def candidate_grid() -> list[dict]:
    grid = [{"reference_samples": reference, "exponent": exponent} for reference in REFERENCE_GRID for exponent in EXPONENT_GRID]
    validate_grid(grid)
    return grid


def validate_grid(grid: object) -> None:
    expected = {(reference, exponent) for reference in REFERENCE_GRID for exponent in EXPONENT_GRID}
    if not isinstance(grid, list) or len(grid) != 338:
        raise ValueError("Calibration grid must contain all 338 predefined candidates")
    actual = []
    for policy in grid:
        validated = scaling.validate_scaling_policy(policy)
        actual.append((validated["reference_samples"], validated["exponent"]))
    if len(set(actual)) != len(actual) or set(actual) != expected or (600, .5) not in actual:
        raise ValueError("Calibration grid has missing, unexpected, or duplicate candidates")


def _means(rows: list[dict], seeds) -> dict[tuple[int, str], float]:
    return {(count, profile): math.fsum(row["validation_rmse_km"] for row in rows
                                      if row["train_count"] == count and row["profile"] == profile) / len(seeds)
            for count in COUNTS for profile in PROFILES}


def _score(policy: dict, means: dict) -> dict:
    assignments = []
    reference, exponent = policy["reference_samples"], policy["exponent"]
    medium_parameters = scaling.parameter_count("medium", BASE_CHANNELS)
    logs = {profile: math.log(reference) + math.log(scaling.parameter_count(profile, BASE_CHANNELS) / medium_parameters) / exponent
            for profile in PROFILES[1:]}
    for count in COUNTS:
        resolved = scaling.resolve_capacity("auto", train_count=count, base_channels=BASE_CHANNELS, scaling_policy=policy)
        profile = resolved["selected_profile"]
        best = min(PROFILES, key=lambda name: (means[count, name], scaling.parameter_count(name, BASE_CHANNELS)))
        assignments.append({"train_count": count, "profile": profile, "parameter_count": resolved["parameter_count"],
                            "mean_validation_rmse_km": means[count, profile], "oracle_profile": best,
                            "oracle_mean_validation_rmse_km": means[count, best],
                            "relative_regret": means[count, profile] / means[count, best] - 1.})
    return {"policy": dict(policy), "score": math.fsum(row["relative_regret"] for row in assignments) / len(COUNTS),
            "mean_parameter_count": math.fsum(row["parameter_count"] for row in assignments) / len(COUNTS),
            "threshold_margin": min(abs(math.log(count) - threshold) for count in COUNTS for threshold in logs.values()),
            "thresholds": {profile: math.exp(value) for profile, value in logs.items()}, "assignments": assignments}


def score_policy(rows: list[dict], policy: dict, *, seeds=DEVELOPMENT_SEEDS) -> dict:
    rows = validate_projection(rows, seeds=seeds)
    return _score(scaling.validate_scaling_policy(policy), _means(rows, seeds))


def choose_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("Cannot select from empty candidate scores")
    best_score = min(row["score"] for row in candidates)
    finalists = [row for row in candidates if row["score"] <= best_score + TOLERANCE]
    lowest_parameters = min(row["mean_parameter_count"] for row in finalists)
    finalists = [row for row in finalists if row["mean_parameter_count"] == lowest_parameters]
    groups = {}
    for row in finalists:
        mapping = tuple(item["profile"] for item in row["assignments"])
        groups.setdefault(mapping, []).append(row)
    representatives = []
    for group in groups.values():
        largest_margin = max(row["threshold_margin"] for row in group)
        representatives.extend(row for row in group if row["threshold_margin"] == largest_margin)
    return min(representatives, key=lambda row: (row["policy"]["exponent"], row["policy"]["reference_samples"]))


def fit_policy(rows: list[dict], *, seeds=DEVELOPMENT_SEEDS) -> tuple[dict, list[dict]]:
    rows = validate_projection(rows, seeds=seeds)
    means = _means(rows, seeds)
    candidates = [_score(policy, means) for policy in candidate_grid()]
    return choose_candidate(candidates), candidates


def leave_one_seed_out(rows: list[dict]) -> list[dict]:
    rows = validate_projection(rows)
    folds = []
    for held_out in DEVELOPMENT_SEEDS:
        training_seeds = [seed for seed in DEVELOPMENT_SEEDS if seed != held_out]
        training = [row for row in rows if row["seed"] in training_seeds]
        held_out_rows = [row for row in rows if row["seed"] == held_out]
        fitted, _ = fit_policy(training, seeds=training_seeds)
        folds.append({"held_out_seed": held_out, "fit_seeds": training_seeds, "policy": fitted["policy"],
                      "fit_score": fitted["score"], "fit_assignments": fitted["assignments"],
                      "held_out": score_policy(held_out_rows, fitted["policy"], seeds=[held_out]),
                      "held_out_original_rule": score_policy(held_out_rows, OLD_POLICY, seeds=[held_out])})
    return folds


def calibrate(analysis_path: Path, output_dir: Path, *, study_path: Path | None = None) -> dict:
    rows, source = load_development_projection(analysis_path, study_path=study_path)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("Calibration output must be a new directory")
    if any(Path(item["path"]).resolve().is_relative_to(output_dir.resolve()) for item in source.values() if isinstance(item, dict)):
        raise ValueError("Calibration output cannot contain its source evidence")
    selected, candidates = fit_policy(rows)
    loso = leave_one_seed_out(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "candidate-scores.csv"
    columns = ["reference_samples", "exponent", "score", "mean_parameter_count", "threshold_margin", "selected"]
    columns.extend(f"profile_at_{count}" for count in COUNTS)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({**candidate["policy"], **{key: candidate[key] for key in ("score", "mean_parameter_count", "threshold_margin")},
                             "selected": candidate["policy"] == selected["policy"],
                             **{f"profile_at_{item['train_count']}": item["profile"] for item in candidate["assignments"]}})
    artifact = {
        "schema_version": 1, "kind": "validation-only-capacity-calibration", "policy_id": "validation-calibrated-v2",
        "created_at": datetime.now(timezone.utc).isoformat(), "method": "SCM", "base_channels": BASE_CHANNELS,
        "development_seeds": list(DEVELOPMENT_SEEDS), "policy": selected["policy"], "source": source,
        "projection_sha256": study.digest(rows), "projection": rows,
        "objective": {"name": "equal-N-weighted mean relative validation-RMSE regret",
                      "formula": "mean_N(mean_seed(validation_rmse[N, policy(N)]) / min_profile mean_seed(validation_rmse[N, profile]) - 1)",
                      "N_weights": "equal", "validation_checkpoint": "first minimum validation RMSE of each original training run",
                      "oracle_profiles": list(PROFILES), "test_metrics_used": False},
        "grid": {"reference_samples": list(REFERENCE_GRID), "exponents": list(EXPONENT_GRID), "candidate_count": len(candidates),
                 "original_rule_included": True, "candidate_scores": {"path": csv_path.name, "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}},
        "tie_rule": TIE_RULE, "assignments": selected["assignments"], "score": selected["score"],
        "mean_parameter_count": selected["mean_parameter_count"], "threshold_margin": selected["threshold_margin"],
        "thresholds": selected["thresholds"], "original_rule": score_policy(rows, OLD_POLICY), "leave_one_seed_out": loso,
        "bounds": {"observed_train_counts": list(COUNTS), "train_count_min": min(COUNTS), "train_count_max": max(COUNTS),
                   "reference_samples_min": min(REFERENCE_GRID), "reference_samples_max": max(REFERENCE_GRID),
                   "exponent_min": min(EXPONENT_GRID), "exponent_max": max(EXPONENT_GRID)},
        "limitations": ["Development validation calibration; this is not independent confirmation of predictive performance.",
                        "Same-record full_interleaved windows are temporally correlated and do not establish cross-environment generalization.",
                        "Leave-one-development-seed-out results are descriptive sensitivity analyses, not confidence intervals or significance tests.",
                        "Seed changes include split and optimizer randomness; three seeds are not three independent environments.",
                        "Only the four fixed base-16 profiles are eligible; auto and baseline observations never enter fitting.",
                        "Candidate grid and four observed training counts bound this calibration; interpolation or extrapolation is unvalidated.",
                        "Matched fixed-profile observations used to evaluate a policy are reused observations, not additional training replicates.",
                        "The default training-count rule is unchanged; applying this artifact requires an explicit separately frozen evaluation plan."],
        "implementation": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for path in (Path(__file__), Path(scaling.__file__))],
    }
    (output_dir / ARTIFACT_NAME).write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return artifact


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--analysis", required=True, type=Path, help="Verified original analysis/study-results.json")
    parser.add_argument("--study", type=Path, help="Source study.json; defaults to analysis directory's parent")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for artifact and all candidate scores")
    args = parser.parse_args(argv)
    try:
        result = calibrate(args.analysis, args.output_dir, study_path=args.study)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Calibration failed: {exc}\n")
    print(json.dumps({"status": "PASS", "artifact": str(args.output_dir / ARTIFACT_NAME),
                      "policy": result["policy"], "score": result["score"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
