"""Plot saved stage metrics/predictions; run rendering only on haiqin."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _count(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def _read_predictions(path):
    required = {"source_segment_idx", "window_center_s", "true_range_km", "pred_range_km", "error_m"}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(f"Incomplete prediction columns: {path}")
        values = []
        for row in reader:
            source = int(row["source_segment_idx"])
            if source <= 0:
                raise ValueError("Prediction source IDs must be positive")
            time = None if row["window_center_s"] == "" else float(row["window_center_s"])
            truth, prediction, error = (float(row[key]) for key in ("true_range_km", "pred_range_km", "error_m"))
            for name, value in (("truth", truth), ("prediction", prediction), ("error", error)):
                _number(value, name)
            if time is not None:
                _number(time, "window_center_s")
            if not math.isclose(error, 1000 * (prediction - truth), rel_tol=1e-6, abs_tol=1e-5):
                raise ValueError(f"Prediction error column is inconsistent: {path}")
            values.append({"source_id": source, "time_s": time, "truth_km": truth,
                           "prediction_km": prediction, "error_m": error})
    if not values or len({item["source_id"] for item in values}) != len(values):
        raise ValueError(f"Prediction source identities must be nonempty and unique: {path}")
    return values


def _saved_data(report_path):
    """Validate identities and saved metrics before importing plotting libraries."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("pack") != "MoE_learn" or report.get("status") != "PASS":
        raise ValueError("Stage figures require a completed MoE_learn report")
    stages = report.get("stages")
    if not isinstance(stages, list) or len(stages) < 2:
        raise ValueError("Stage figures require at least two deployed stages")
    sources = [report_path]
    predictions, summaries = [], []
    baseline, seen_rows, previous_after = None, 0, 0
    for number, stage in enumerate(stages, 1):
        if stage.get("stage") != number:
            raise ValueError("Stage records must be consecutive and one based")
        _count(stage["experts"], "experts")
        new_rows = _count(stage["new_training_rows"], "new_training_rows")
        seen_rows += new_rows
        if _count(stage["cumulative_training_rows"], "cumulative_training_rows") != seen_rows:
            raise ValueError("Cumulative unique training rows disagree with the stage counts")
        after = _count(stage["cumulative_after_training_rows"], "cumulative_after_training_rows", zero=True)
        if after < previous_after or after > seen_rows or (number == 1 and after != 0):
            raise ValueError("After-side training counts must start at zero and be cumulative")
        previous_after = after
        if not isinstance(stage.get("action"), str) or not stage["action"]:
            raise ValueError("Missing deployed stage action")
        path = report_path.parent / f"predictions-stage-{number:03d}.csv"
        observed = _read_predictions(path)
        identity = [(item["source_id"], item["time_s"], item["truth_km"]) for item in observed]
        if baseline is None:
            baseline = identity
            metadata_ids = report.get("metadata", {}).get("test_source_ids")
            if metadata_ids is not None and metadata_ids != [item[0] for item in identity]:
                raise ValueError("Predictions disagree with the saved fixed test membership")
        elif identity != baseline:
            raise ValueError("Stage predictions do not use the same test identities, times and truths")
        identity_hash = hashlib.sha256(",".join(str(item[0]) for item in identity).encode()).hexdigest()
        if stage.get("test_source_identity_sha256", identity_hash) != identity_hash:
            raise ValueError("Stage test source digest is inconsistent")
        metrics = stage.get("test")
        if not isinstance(metrics, dict) or metrics.get("count") != len(observed):
            raise ValueError("Saved test metrics must cover every exported test prediction")
        errors = [item["error_m"] for item in observed]
        computed = {"mae_km": math.fsum(abs(value) for value in errors) / len(errors) / 1000,
                    "rmse_km": math.sqrt(math.fsum(value * value for value in errors) / len(errors)) / 1000}
        for key, actual in computed.items():
            stated = _number(metrics[key], f"test.{key}", nonnegative=True)
            if not math.isclose(actual, stated, rel_tol=2e-5, abs_tol=1e-6):
                raise ValueError(f"Stage {number} report {key} disagrees with its prediction CSV")
        validation = stage.get("validation_by_stage")
        if not isinstance(validation, list):
            raise ValueError("Missing per-stage validation record")
        first_validation = [item for item in validation if item.get("stage") == 1]
        if len(first_validation) > 1:
            raise ValueError("Duplicate stage-one validation metrics")
        before_rmse = None
        if first_validation:
            _count(first_validation[0]["count"], "stage-one validation count")
            before_rmse = 1000 * _number(first_validation[0]["rmse_km"], "stage-one validation RMSE", nonnegative=True)
        summaries.append({"stage": number, "action": stage["action"], "experts": stage["experts"],
                          "new_training_rows": new_rows, "cumulative_training_rows": seen_rows,
                          "cumulative_after_training_rows": after, "test_count": len(observed),
                          "test_mae_m": 1000 * metrics["mae_km"], "test_rmse_m": 1000 * metrics["rmse_km"],
                          "stage_one_validation_rmse_m": before_rmse})
        predictions.append(observed)
        sources.append(path)
    return report, summaries, predictions, sources


def _range_group_data(report, stages, predictions):
    """Validate saved distance partitions and derive exact errors from fixed rows."""
    allocation = report.get("metadata", {}).get("allocation", {})
    raw_groups = allocation.get("range_groups")
    if allocation.get("after_grouping") != "range_bins" and (raw_groups is None or raw_groups == []):
        return None
    if not isinstance(raw_groups, list) or not raw_groups or len(raw_groups) != len(stages) - 1:
        raise ValueError("Distance groups must correspond one to one with after-side stages")
    memberships = report["metadata"].get("stages")
    if not isinstance(memberships, list) or len(memberships) != len(stages):
        raise ValueError("Distance figures require each stage's saved source membership")
    groups, positions_seen, sources_seen = [], set(), set()
    previous_upper, previous_inclusive, cumulative = None, False, 0
    for number, group in enumerate(raw_groups, 1):
        if not isinstance(group, dict) or group.get("group") != number:
            raise ValueError("Distance group identities must be consecutive and one based")
        lower = _number(group["lower_km"], "lower_km", nonnegative=True)
        upper = _number(group["upper_km"], "upper_km", nonnegative=True)
        inclusive = group["upper_inclusive"]
        if type(inclusive) is not bool or lower >= upper:
            raise ValueError("Distance groups require increasing bounds and explicit upper inclusion")
        if previous_upper is not None and (lower < previous_upper or (lower == previous_upper and previous_inclusive)):
            raise ValueError("Distance group intervals overlap or are out of order")
        previous_upper, previous_inclusive = upper, inclusive
        positions = group["test_positions"]
        if (not isinstance(positions, list) or not positions
                or any(type(index) is not int or not 0 <= index < len(predictions[0]) for index in positions)
                or len(set(positions)) != len(positions) or positions_seen.intersection(positions)):
            raise ValueError("Distance test positions must be nonempty, unique and disjoint")
        positions_seen.update(positions)
        for role in ("train", "val", "test"):
            ids = group[f"{role}_source_ids"]
            count = _count(group[f"{role}_count"], f"{role}_count")
            if (not isinstance(ids, list) or len(ids) != count or len(set(ids)) != count
                    or any(type(source) is not int or source <= 0 for source in ids)
                    or sources_seen.intersection(ids)):
                raise ValueError("Distance group source roles must be unique, disjoint and match counts")
            sources_seen.update(ids)
            if role != "test" and ids != memberships[number].get(f"{role}_source_ids"):
                raise ValueError("Distance group membership differs from its incremental training stage")
        if (len(positions) != group["test_count"]
                or [predictions[0][index]["source_id"] for index in positions] != group["test_source_ids"]):
            raise ValueError("Distance test positions differ from saved test source identities")
        raw_ranges = group.get("test_ranges_km")
        if not isinstance(raw_ranges, list) or len(raw_ranges) != len(positions):
            raise ValueError("Distance groups require raw test ranges aligned with test positions")
        for index, raw_range in zip(positions, raw_ranges):
            raw_range = _number(raw_range, "test_ranges_km", nonnegative=True)
            if not (lower <= raw_range and (raw_range <= upper if inclusive else raw_range < upper)):
                raise ValueError("A raw fixed test range falls outside its declared distance group")
            # The loader casts labels to float32; membership uses the unrounded
            # source range, so a rounded label cannot move an observation across bins.
            truth = predictions[0][index]["truth_km"]
            if not math.isclose(raw_range, truth, rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError("A prediction CSV truth differs from its raw test range beyond float32 precision")
        cumulative += group["train_count"]
        if (stages[number]["new_training_rows"] != group["train_count"]
                or stages[number]["cumulative_after_training_rows"] != cumulative):
            raise ValueError("Distance group counts differ from deployed stage training counts")
        groups.append({**group, "lower_km": lower, "upper_km": upper})
    if positions_seen != set(range(len(predictions[0]))):
        raise ValueError("Distance groups must cover every fixed test row exactly once")
    before_ids = set(memberships[0]["train_source_ids"] + memberships[0]["val_source_ids"])
    if before_ids.intersection(sources_seen):
        raise ValueError("Distance group sources overlap the initial before-side sources")
    rows = []
    for stage, observed in zip(stages, predictions):
        for group in [None, *groups]:
            positions = range(len(observed)) if group is None else group["test_positions"]
            errors = [1000 * (observed[index]["prediction_km"] - observed[index]["truth_km"]) for index in positions]
            rows.append({"stage": stage["stage"], "action": stage["action"], "experts": stage["experts"],
                         "cumulative_after_training_rows": stage["cumulative_after_training_rows"],
                         "introduced_after_group": 0 if stage["stage"] == 1 else stage["stage"] - 1,
                         "group": 0 if group is None else group["group"],
                         "scope": "overall" if group is None else "distance_group",
                         "lower_km": "" if group is None else group["lower_km"],
                         "upper_km": "" if group is None else group["upper_km"],
                         "upper_inclusive": "" if group is None else group["upper_inclusive"],
                         "test_count": len(errors), "mae_m": math.fsum(abs(value) for value in errors) / len(errors),
                         "rmse_m": math.sqrt(math.fsum(value * value for value in errors) / len(errors))})
    return groups, rows


def _interval(group):
    return f"[{group['lower_km']:.6g}, {group['upper_km']:.6g}{']' if group['upper_inclusive'] else ')'} km"


def _distance_holdout_data(report, stages, predictions):
    """Audit a complete distance holdout using raw ranges, before rendering."""
    metadata = report.get("metadata", {})
    if metadata.get("stage_mode") != "distance_holdout":
        return None
    allocation = metadata.get("allocation", {})
    if allocation.get("range_groups") not in (None, []):
        raise ValueError("A complete distance holdout cannot use incremental test range groups")
    if allocation.get("test_range_excluded_from_learning") != "PASS":
        raise ValueError("Missing complete test-distance exclusion certification")
    edges = allocation.get("distance_edges_km")
    if not isinstance(edges, list) or len(edges) < 3:
        raise ValueError("Distance holdout requires at least two predeclared distance groups")
    edges = [_number(value, "distance_edges_km", nonnegative=True) for value in edges]
    if any(left >= right for left, right in zip(edges, edges[1:])):
        raise ValueError("Distance holdout edges must be strictly increasing")
    raw_groups = allocation.get("distance_groups")
    if not isinstance(raw_groups, list) or len(raw_groups) != len(edges) - 1:
        raise ValueError("Distance holdout must describe every predeclared distance group")
    groups = []
    for number, group in enumerate(raw_groups, 1):
        if not isinstance(group, dict) or type(group.get("group")) is not int or group["group"] != number:
            raise ValueError("Distance holdout group IDs must be consecutive and one based")
        lower = _number(group.get("lower_km"), "lower_km", nonnegative=True)
        upper = _number(group.get("upper_km"), "upper_km", nonnegative=True)
        inclusive = group.get("upper_inclusive")
        if (lower != edges[number - 1] or upper != edges[number]
                or type(inclusive) is not bool or inclusive != (number == len(raw_groups))):
            raise ValueError("Distance group bounds must match the declared disjoint edges")
        groups.append({**group, "lower_km": lower, "upper_km": upper})
    heldout_number = _count(allocation.get("heldout_group"), "heldout_group")
    if heldout_number > len(groups):
        raise ValueError("Held-out distance group is out of range")
    heldout = groups[heldout_number - 1]

    def within(value, group):
        return (group["lower_km"] <= value
                and (value <= group["upper_km"] if group["upper_inclusive"] else value < group["upper_km"]))

    def source_ids(values, name, *, allow_empty=False):
        if (not isinstance(values, list) or (not values and not allow_empty)
                or any(type(value) is not int or value <= 0 for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{name} must contain unique positive source IDs")
        return values

    def ranges(values, expected, name):
        if not isinstance(values, list) or len(values) != expected:
            raise ValueError(f"{name} must align with source IDs")
        return [_number(value, name, nonnegative=True) for value in values]

    full_pool_ids = {"before": [], "after": []}
    for group in groups:
        for side in ("before", "after"):
            ids = source_ids(group.get(f"{side}_source_ids"), f"group {group['group']} {side}_source_ids", allow_empty=True)
            values = ranges(group.get(f"{side}_ranges_km"), len(ids), f"group {group['group']} {side}_ranges_km")
            if any(not within(value, group) for value in values):
                raise ValueError("A complete source-pool range falls outside its declared distance group")
            if _count(group.get(f"{side}_count"), f"{side}_count", zero=True) != len(ids):
                raise ValueError("Complete distance group counts disagree with source memberships")
            full_pool_ids[side].extend(ids)
    full_ids = full_pool_ids["before"] + full_pool_ids["after"]
    if len(set(full_ids)) != len(full_ids):
        raise ValueError("Complete before/after distance groups must have disjoint source IDs")
    test_ids = source_ids(metadata.get("test_source_ids"), "test_source_ids")
    if test_ids != [row["source_id"] for row in predictions[0]]:
        raise ValueError("Distance holdout membership differs from the prediction CSVs")
    if test_ids != heldout["after_source_ids"]:
        raise ValueError("The fixed test must contain the complete held-out after-side distance group")
    test_ranges = ranges(metadata.get("test_ranges_km"), len(test_ids), "test_ranges_km")
    if test_ranges != heldout["after_ranges_km"]:
        raise ValueError("Raw test ranges disagree with the complete held-out distance group")
    if metadata.get("test_source_side") != "after":
        raise ValueError("Distance holdout test must come from the after-side pool")
    for value, prediction in zip(test_ranges, predictions[0]):
        if not within(value, heldout):
            raise ValueError("A raw fixed-test range falls outside the held-out distance group")
        if not math.isclose(value, prediction["truth_km"], rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError("A prediction truth differs from its raw held-out range beyond float32 precision")
    memberships = metadata.get("stages")
    if not isinstance(memberships, list) or len(memberships) != len(groups) or len(stages) != len(groups):
        raise ValueError("Distance holdout requires one before stage and one stage per retained after group")
    after_groups = [group for group in groups if group["group"] != heldout_number]
    sources_seen, checked_stages, cumulative_after = set(test_ids), [], 0
    for number, (membership, stage) in enumerate(zip(memberships, stages), 1):
        side = "before" if number == 1 else "after"
        if membership.get("stage") != number or membership.get("source_side") != side:
            raise ValueError("Distance holdout stages must start with before, followed by after")
        expected_group = None if number == 1 else after_groups[number - 2]
        if expected_group is not None:
            stated = membership.get("range_group")
            if (not isinstance(stated, dict)
                    or any(type(stated.get(key)) is not type(expected_group[key])
                           or stated[key] != expected_group[key]
                           for key in ("group", "upper_inclusive"))
                    or any(_number(stated.get(key), key) != expected_group[key] for key in ("lower_km", "upper_km"))):
                raise ValueError("After stages must introduce non-held-out groups from near to far")
        checked = {"stage": number, "source_side": side, "range_group": expected_group}
        for role in ("train", "val"):
            ids = source_ids(membership.get(f"{role}_source_ids"), f"stage {number} {role}_source_ids")
            values = ranges(membership.get(f"{role}_ranges_km"), len(ids), f"stage {number} {role}_ranges_km")
            if sources_seen.intersection(ids):
                raise ValueError("Distance holdout test, training and validation source IDs must be disjoint")
            if not set(ids).issubset(full_pool_ids[side]):
                raise ValueError("A stage source ID does not belong to its declared before/after pool")
            if expected_group is not None and not set(ids).issubset(expected_group["after_source_ids"]):
                raise ValueError("An after-stage source ID does not belong to its declared distance group")
            sources_seen.update(ids)
            if any(within(value, heldout) for value in values):
                raise ValueError("A training or validation range falls in the held-out distance group")
            if expected_group is not None and any(not within(value, expected_group) for value in values):
                raise ValueError("An after-side learning range falls outside its declared distance group")
            if _count(membership.get(f"{role}_count"), f"{role}_count") != len(ids):
                raise ValueError("Distance holdout stage counts disagree with source memberships")
            checked[f"{role}_source_ids"], checked[f"{role}_ranges_km"] = ids, values
        if len(checked["train_source_ids"]) != stage["new_training_rows"]:
            raise ValueError("Distance holdout training membership disagrees with the deployed stage count")
        if number > 1:
            cumulative_after += len(checked["train_source_ids"])
        if stage["cumulative_after_training_rows"] != cumulative_after:
            raise ValueError("Distance holdout cumulative after counts disagree with stage memberships")
        checked_stages.append(checked)
    excluded = source_ids(allocation.get("before", {}).get("excluded_source_ids"),
                          "before.excluded_source_ids", allow_empty=True)
    if sources_seen.intersection(excluded):
        raise ValueError("Excluded before-side rows cannot appear in learning or test roles")
    excluded_ranges = ranges(allocation["before"].get("excluded_ranges_km"), len(excluded), "before.excluded_ranges_km")
    if (excluded != heldout["before_source_ids"] or excluded_ranges != heldout["before_ranges_km"]
            or any(not within(value, heldout) for value in excluded_ranges)):
        raise ValueError("Before-side exclusions must contain the complete held-out distance group")
    if sources_seen.union(excluded) != set(full_ids):
        raise ValueError("Learning, test and before-side exclusions must cover every source-pool row")
    if metadata.get("preprocessing_fit_stage") != 1:
        raise ValueError("Distance holdout preprocessing must be fitted on the retained initial stage")
    return {
        "heldout_group": heldout_number, "heldout_interval": heldout,
        "distance_edges_km": edges, "distance_groups": groups,
        "before_excluded_source_ids": excluded, "before_excluded_ranges_km": excluded_ranges,
        "before_excluded_count": len(excluded),
        "test_source_ids": test_ids, "test_ranges_km": test_ranges, "stage_memberships": checked_stages,
        "membership_check": "PASS: all test ranges are in the held-out group; every stage train/validation range is outside it; all source roles and excluded before IDs are disjoint",
        "distance_boundary_precision": "membership uses raw source ranges and exact interval bounds; CSV truths match raw ranges within rel_tol=1e-7/abs_tol=1e-8 for float32 conversion",
        "test_role": "fixed after-side reporting only; test metrics do not select or train candidates",
        "interpretation": "the held-out distance is excluded from before and after training/validation; other after groups are introduced from near to far",
    }


def _write_distance_outputs(plt, temporary, groups, rows, stages, dpi):
    """Render grouped holdout errors without implying that before lacked these ranges."""
    with (temporary / "range-groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    width, height = 12.8, 8.6
    fig = plt.figure(figsize=(width, height), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=[1, .22, .50, .20])
    axes = [fig.add_subplot(grid[0, column]) for column in (0, 1)]
    legend_ax, table_ax, note_ax = (fig.add_subplot(grid[index, :]) for index in (1, 2, 3))
    fig.suptitle("Continual MoE: fixed after-side test by distance group", fontsize=15)
    colors = ["#0072B2", "#D55E00", "#009E73", "#882255", "#332288", "#117733", "#AA4499"]
    markers, linestyles = ["o", "s", "^", "v", "P", "X", "d"], ["-", "--", "-.", ":"]
    x = [stage["cumulative_after_training_rows"] for stage in stages]
    ymax = 1.12 * max(row[key] for row in rows for key in ("mae_m", "rmse_m")) or 1.0
    try:
        for axis, key, title in zip(axes, ("mae_m", "rmse_m"), ("(a) Fixed test MAE", "(b) Fixed test RMSE")):
            for index, group in enumerate(groups):
                values = [row[key] for row in rows if row["group"] == group["group"]]
                axis.plot(x, values, color=colors[index % len(colors)], marker=markers[index % len(markers)],
                          linestyle=linestyles[index % len(linestyles)], linewidth=1.6, markersize=6,
                          label=f"G{group['group']} {_interval(group)}; n={group['test_count']}")
            axis.plot(x, [row[key] for row in rows if row["group"] == 0], color="#222222", marker="D",
                      linestyle=":", linewidth=2, markersize=6, label=f"Overall; n={sum(group['test_count'] for group in groups)}")
            axis.set(title=title, xlabel="Stage / cumulative after-side training rows", ylabel="Range error (m)",
                     xticks=x, xticklabels=[f"S{stage['stage']}\n{stage['cumulative_after_training_rows']}" for stage in stages],
                     ylim=(0, ymax))
            axis.grid(alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
        legend_ax.axis("off")
        legend_ax.legend(*axes[0].get_legend_handles_labels(), loc="center", ncol=2, frameon=False, fontsize=9)
        table_ax.axis("off")
        table_ax.set_title("After-side training rows introduced at each stage", loc="left")
        table = table_ax.table(
            cellText=[[stage["stage"], "Before only" if number == 0 else f"G{number} {_interval(groups[number - 1])}",
                       0 if number == 0 else groups[number - 1]["train_count"], stage["action"]]
                      for number, stage in enumerate(stages)],
            colLabels=["Stage", "Incoming after-side distance interval", "New after rows", "Deployed action"],
            colWidths=[.07, .43, .15, .30], cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#D0D0D0")
            cell.set_linewidth(.5)
            if row == 0:
                cell.set_facecolor("#EBEFF3")
                cell.get_text().set_weight("bold")
        note_ax.axis("off")
        note_ax.text(0, .98,
                     "Identical fixed test rows at every stage; test errors are reporting only. Single run; no uncertainty intervals.\n"
                     "Before-side training may already contain these distances; a group not yet added from after is not necessarily unseen.\n"
                     "Lines connect deployed stages only; no smoothing. Counts exclude replay repetitions; overall errors pool individual test rows.",
                     va="top", fontsize=9, transform=note_ax.transAxes)
        for suffix in ("png", "pdf"):
            fig.savefig(temporary / f"distance-learning.{suffix}", format=suffix, dpi=dpi, facecolor="white")
    finally:
        plt.close(fig)
    return {"width_in": width, "height_in": height, "png_dpi": dpi, "pdf_fonttype": 42, "background": "white"}


def plot_stages(report_path, output_dir):
    """Write stage-learning.png/.pdf and provenance from immutable saved values.

    The caller runs this on haiqin after the PASS report and prediction CSVs
    exist. Use a new output directory to rerender; existing outputs are refused.
    """
    report_path, output_dir = Path(report_path).resolve(), Path(output_dir).resolve()
    report, stages, predictions, sources = _saved_data(report_path)
    holdout = _distance_holdout_data(report, stages, predictions)
    grouped = _range_group_data(report, stages, predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("stage-learning.png", "stage-learning.pdf", "stage-learning-provenance.json")
    if grouped is not None:
        names += ("range-groups.csv", "distance-learning.png", "distance-learning.pdf")
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("Stage figures already exist; select a new output directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    after_counts = [row["cumulative_after_training_rows"] for row in stages]
    use_after = any(after_counts)
    x = after_counts if use_after else [row["stage"] for row in stages]
    x_label = "Cumulative after-side training rows" if use_after else "Deployment stage"
    before_mode = report.get("metadata", {}).get("stage_mode") in {"before_then_after", "distance_holdout"}
    first_label = "Before-side validation RMSE" if before_mode else "Stage-one validation RMSE"
    if holdout is not None:
        first_label = "Before-side validation (retained distances)"
    all_times = all(row["time_s"] is not None for row in predictions[0])
    time_key, time_label = ("time_s", "Window center time (s)") if all_times else ("source_id", "Source segment ID (timestamps unavailable)")
    order = sorted(range(len(predictions[0])), key=lambda index: predictions[0][index][time_key])
    styles = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
              "axes.labelsize": 10, "legend.fontsize": 9, "pdf.fonttype": 42,
              "savefig.facecolor": "white", "figure.facecolor": "white"}
    width, height, dpi = 12.8, 10.8 if holdout is not None else 10.4, 300
    with plt.rc_context(styles):
        fig = plt.figure(figsize=(width, height), layout="constrained")
        grid = fig.add_gridspec(4, 2, height_ratios=[1, 1.18, .62, .26 if holdout is not None else .15])
        test_ax, validation_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
        prediction_ax = fig.add_subplot(grid[1, :])
        table_ax, note_ax = fig.add_subplot(grid[2, :]), fig.add_subplot(grid[3, :])
        title = "Continual MoE: fixed after-side test" if before_mode else "Continual MoE: fixed test"
        if holdout is not None:
            title = f"Continual MoE: after-side distance holdout G{holdout['heldout_group']} {_interval(holdout['heldout_interval'])}"
        fig.suptitle(title, fontsize=15)
        test_ax.plot(x, [row["test_mae_m"] for row in stages], color="#0072B2", marker="o", label="Test MAE")
        test_ax.plot(x, [row["test_rmse_m"] for row in stages], color="#D55E00", marker="s", linestyle="--", label="Test RMSE")
        test_ax.set(title="(a) Fixed test error", xlabel=x_label, ylabel="Range error (m)", xticks=x)
        test_ax.legend(frameon=False)
        validation_y = [float("nan") if row["stage_one_validation_rmse_m"] is None else row["stage_one_validation_rmse_m"] for row in stages]
        validation_ax.plot(x, validation_y, color="#333333", marker="D", linestyle="-.")
        validation_ax.set(title=f"(b) {first_label}", xlabel=x_label, ylabel="Range RMSE (m)", xticks=x)
        missing_validation = [row["stage"] for row in stages if row["stage_one_validation_rmse_m"] is None]
        if missing_validation:
            validation_ax.text(.02, .96, "Missing validation: stages " + ", ".join(map(str, missing_validation)),
                               transform=validation_ax.transAxes, va="top", fontsize=8)
        for axis in (test_ax, validation_ax):
            finite_values = [float(value) for line in axis.lines for value in line.get_ydata()
                             if math.isfinite(float(value))]
            axis.set_ylim(0, 1.12 * max(finite_values, default=1.0) or 1.0)
            axis.grid(alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
        coordinate = [predictions[0][index][time_key] for index in order]
        prediction_ax.scatter(coordinate, [1000 * predictions[0][index]["truth_km"] for index in order],
                              s=13, color="#222222", marker=".", label="Reference range", zorder=3)
        prediction_ax.scatter(coordinate, [1000 * predictions[0][index]["prediction_km"] for index in order],
                              s=24, facecolors="none", edgecolors="#0072B2", marker="o", linewidths=.9,
                              label=f"Initial prediction (stage {stages[0]['stage']})")
        prediction_ax.scatter(coordinate, [1000 * predictions[-1][index]["prediction_km"] for index in order],
                              s=23, color="#D55E00", marker="x", linewidths=.9,
                              label=f"Final prediction (stage {stages[-1]['stage']})")
        prediction_ax.set(title="(c) Initial and final predictions on identical test windows", xlabel=time_label, ylabel="Range (m)")
        prediction_ax.legend(loc="upper center", bbox_to_anchor=(.5, -.14), ncol=3, frameon=False)
        prediction_ax.grid(alpha=.2)
        prediction_ax.spines[["top", "right"]].set_visible(False)
        table_ax.axis("off")
        table_ax.set_title("(d) Deployed expert count and decisions", loc="left")
        table = table_ax.table(
            cellText=[[row["stage"], row["new_training_rows"], row["cumulative_training_rows"],
                       row["cumulative_after_training_rows"], row["experts"], row["action"]] for row in stages],
            colLabels=["Stage", "New training", "Cumulative training", "Cumulative after", "Experts", "Action"],
            colWidths=[.07, .13, .17, .17, .09, .32], cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.55)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#D0D0D0")
            cell.set_linewidth(.5)
            if row == 0:
                cell.set_facecolor("#EBEFF3")
                cell.get_text().set_weight("bold")
        note_ax.axis("off")
        note = (f"Single run; {len(predictions[0])} fixed test windows; no uncertainty intervals. "
                "Test metrics are reporting only.\n"
                "Training counts exclude replay repetitions. Prediction points are not connected; no smoothing or interpolation.")
        if holdout is not None:
            note = (f"Single fold; {len(predictions[0])} fixed after-side test windows; no uncertainty intervals. Test metrics never select candidates.\n"
                    "The test distance interval is excluded from all before/after training and validation stages.\n"
                    f"Before-side rows in that interval were excluded (n={holdout['before_excluded_count']}). "
                    "Counts exclude replay; prediction points are not connected or smoothed.")
        note_ax.text(0, .95, note,
                     va="top", fontsize=9, transform=note_ax.transAxes)
        try:
            with tempfile.TemporaryDirectory(prefix=".stage-learning-", dir=output_dir) as temporary:
                temporary = Path(temporary)
                for suffix in ("png", "pdf"):
                    fig.savefig(temporary / f"stage-learning.{suffix}", format=suffix, dpi=dpi, facecolor="white")
                distance_export = None if grouped is None else _write_distance_outputs(plt, temporary, *grouped, stages, dpi)
                provenance = {
                    "schema_version": 1, "pack": "MoE_learn", "status": "PASS",
                    "scope": "saved-stage-metrics-and-predictions-only", "scientific_performance_evidence": False,
                    "source_files": [{"path": str(path), "size": path.stat().st_size, "sha256": _hash(path)} for path in sources],
                    "plot_program": {"path": str(Path(__file__).resolve()), "sha256": _hash(__file__)},
                    "matplotlib_version": matplotlib.__version__,
                    "x_axis": "cumulative_after_training_rows" if use_after else "stage",
                    "x_axis_fallback_reason": None if use_after else "all cumulative after-side counts are zero",
                    "prediction_x_axis": time_key,
                    "missing_time_count": sum(row["time_s"] is None for row in predictions[0]),
                    "missing_stage_one_validation": missing_validation,
                    "fixed_test_rows": len(predictions[0]), "stage_summary": stages,
                    "transformations": ["convert ranges/errors from km to m", "sort prediction points by displayed x coordinate"],
                    "metric_check": "CSV-derived MAE/RMSE agree with reports within float32 rounding tolerance",
                    "missing_data": "validation is left as gaps; any missing timestamp switches the whole prediction panel to source ID",
                    "uncertainty": "none; one run, no replicate uncertainty estimated",
                    "excluded_observations": 0, "smoothing": "none", "prediction_connections": "none",
                    "export": {"width_in": width, "height_in": height, "png_dpi": dpi, "pdf_fonttype": 42, "background": "white"},
                    "description": "Four panels show fixed-test MAE/RMSE, retained initial-stage validation RMSE, initial/final test predictions and expert decisions.",
                    "files": [{"path": name, "size": (temporary / name).stat().st_size,
                               "sha256": _hash(temporary / name)} for name in names if name != "stage-learning-provenance.json"],
                }
                if grouped is not None:
                    provenance["distance_groups"] = {
                        "groups": grouped[0], "stage_metrics": grouped[1], "export": distance_export,
                        "membership_check": "test positions/source IDs/bounds form an exhaustive disjoint partition; training/validation IDs match stages",
                        "formulas": {"error_m": "1000 * (prediction_km - truth_km)",
                                     "mae_m": "sum(abs(error_m)) / n", "rmse_m": "sqrt(sum(error_m ** 2) / n)"},
                        "overall_aggregation": "all fixed test rows, not an unweighted average of group metrics",
                        "distance_boundary_precision": "raw source ranges checked against unrounded bounds; CSV truths match raw ranges within rel_tol=1e-7/abs_tol=1e-8 for float32 conversion; display bounds use six significant digits",
                        "interpretation": "group g is introduced from after at stage g+1; before-side training may contain the same distances",
                        "description": "Two panels compare MAE and RMSE for each distance group and all fixed test rows; the table identifies each incoming group.",
                    }
                    provenance["transformations"].append("partition identical fixed test rows by saved range-group positions; calculate per-group MAE/RMSE from saved predictions")
                if holdout is not None:
                    provenance["distance_holdout"] = holdout
                    provenance["stage_mode"] = "distance_holdout"
                    provenance["excluded_observations_scope"] = "zero plotted test rows omitted; before-side learning exclusions are recorded separately in distance_holdout"
                    provenance["description"] = "Four panels show errors on one fully held-out after-side distance group, before validation on retained distances, initial/final held-out predictions and deployed decisions."
                (temporary / names[2]).write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                for name in names:
                    (temporary / name).replace(output_dir / name)
        finally:
            plt.close(fig)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = plot_stages(args.report, args.output)
    print(json.dumps({"status": result["status"], "files": result["files"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
