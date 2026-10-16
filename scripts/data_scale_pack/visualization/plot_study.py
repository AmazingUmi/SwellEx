#!/usr/bin/env python3
"""Plot verified data-size/capacity studies on haiqin using the frozen environment."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_scale_pack.localization_networks.workflow.tasks import ARCHITECTURES, METHODS
from scripts.data_scale_pack.visualization.study_results import LIMITATIONS, load_study, validation_selection

LABELS = {
    "complex_CNN": "Auto (chosen before training)",
    "complex_CNN_small": "Small: 3 blocks",
    "complex_CNN_medium": "Medium: 4 blocks",
    "complex_CNN_large": "Large: 6 blocks",
    "complex_CNN_xlarge": "XLarge: 8 blocks",
    "complex_CNN_baseline": "Baseline: 4 original blocks",
}
COLORS = ("#111111", "#0072B2", "#A65300", "#007A5E", "#8E4585", "#B13624")
MARKERS = ("*", "o", "s", "^", "D", "X")
LINES = ("--", "-", "-.", ":", "-", "--")
RENDERER_PATHS = (
    "scripts/data_scale_pack/visualization/plot_study.py",
    "scripts/data_scale_pack/visualization/study_results.py",
    "scripts/data_scale_pack/visualization/results.py",
    "scripts/data_scale_pack/localization_networks/workflow/tasks.py",
    "scripts/data_scale_pack/localization_networks/workflow/config.py",
    "scripts/run_identity.py", "scripts/packs.py",
)


def _csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _renderer_provenance(study: dict, allow_renderer_update: bool) -> tuple[list[dict], dict]:
    source_map = {item["path"]: item for item in study["source_files"]}
    renderer_sources, training_renderer_sources, changed_paths = [], [], []
    for relative in RENDERER_PATHS:
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        training_digest = source_map.get(relative, {}).get("sha256")
        renderer_sources.append({"path": relative, "sha256": digest})
        training_renderer_sources.append({"path": relative, "sha256": training_digest})
        if training_digest != digest:
            changed_paths.append(relative)
    if changed_paths and not allow_renderer_update:
        raise ValueError("Renderer differs from the study's frozen source: " + ", ".join(changed_paths)
                         + "; use --allow-renderer-update for an explicitly recorded redraw")
    return renderer_sources, {
        "training_source_sha256": study["source_sha256"],
        "renderer_source_sha256": hashlib.sha256(json.dumps(renderer_sources, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "training_renderer_sources": training_renderer_sources,
        "matches_training_snapshot": not changed_paths,
        "changed_paths": changed_paths,
        "allow_renderer_update": allow_renderer_update,
        "retrained": False,
        "action": "redraw verified evidence and aggregate recorded metrics; no retraining or prediction",
    }


def render_study(study_path: Path, output_dir: Path, *, runs_root: Path | None = None, dpi: int = 160,
                 allow_renderer_update: bool = False) -> dict:
    if type(dpi) is not int or not 72 <= dpi <= 600:
        raise ValueError("Preview dpi must be an integer between 72 and 600")
    if type(allow_renderer_update) is not bool:
        raise ValueError("allow_renderer_update must be a boolean")
    data = load_study(study_path, runs_root)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("Study output must be a new directory")
    if any(Path(item["path"]).resolve().is_relative_to(output_dir.resolve()) for item in data.sources):
        raise ValueError("Study output cannot contain its evidence inputs")
    renderer_sources, renderer_provenance = _renderer_provenance(data.study, allow_renderer_update)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=False)
    methods = (data.study["plan"]["method"],) if data.study["plan"]["method"] else METHODS
    fractions = sorted(data.study["plan"]["fractions"])
    seeds = data.study["plan"]["seeds"]
    summary = {(row["method"], row["architecture"], row["train_fraction"]): row for row in data.summaries}
    raw = [row for row in data.rows if row["status"] == "PASS"]
    selection = validation_selection(data)
    captions, figures = {}, []
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
             "axes.grid": True, "grid.alpha": 0.18, "axes.spines.top": False,
             "axes.spines.right": False, "pdf.fonttype": 42, "savefig.transparent": False,
             "legend.edgecolor": "#b0b0b0", "legend.framealpha": 1, "legend.fancybox": False}
    note = f"{len(seeds)} planned seeds; raw points + mean ± sample SD; no generalization CI. Missing cells stay NOT_RUN."
    handles = [Line2D([], [], color=COLORS[i], marker=MARKERS[i], linestyle=LINES[i], label=LABELS[name]) for i, name in enumerate(ARCHITECTURES)]

    def panel_grid(rows=1, columns=None, height=4.2, width=5.5):
        columns = len(methods) if columns is None else columns
        fig, axes = plt.subplots(rows, columns, figsize=(max(width * columns, 10), height * rows + 1.7), squeeze=False)
        bottom = 0.22 if rows == 1 else max(0.15, 1.8 / fig.get_figheight())
        fig.subplots_adjust(left=0.09, right=0.97, top=0.90 if rows >= 3 else 0.85,
                            bottom=bottom, wspace=0.27, hspace=0.55 if rows >= 3 else 0.3)
        return fig, axes

    def finish(fig, stem, title, description, *, legend=True, selected_handles=None):
        complete = sum(row["status"] == "PASS" for row in data.rows)
        fig.suptitle(title + "\n" + data.study["study_id"] + f" | {data.status}: {complete}/{len(data.rows)} model-seed cells", y=0.98, fontsize=12)
        fig.text(0.5, 0.025, note, ha="center", va="bottom", fontsize=8)
        if legend:
            fig.legend(handles=handles if selected_handles is None else selected_handles, loc="lower center",
                       bbox_to_anchor=(0.5, 0.065), ncol=3, fontsize=9, frameon=True)
        for extension in ("pdf", "png"):
            path = output_dir / f"{stem}.{extension}"
            fig.savefig(path, format=extension, dpi=dpi, facecolor="white")
            figures.append(path.name)
        captions[stem] = {"title": title, "description": description, "uncertainty": note,
                          "width_inches": fig.get_figwidth(), "height_inches": fig.get_figheight()}
        plt.close(fig)

    def observed(method, architecture, fraction):
        return [row for row in raw if row["method"] == method and row["architecture"] == architecture and row["train_fraction"] == fraction]

    def sample_curve(ax, method, metric, *, xmetric="train_count"):
        for index, architecture in enumerate(ARCHITECTURES):
            xs, ys = [], []
            for fraction in fractions:
                row = summary[(method, architecture, fraction)]
                if not row["n_seeds"]:
                    xs.append(float("nan")); ys.append(float("nan"))
                    continue
                x = row["train_count"] if xmetric == "train_count" else row[xmetric + "_mean"]
                y, error = row[metric + "_mean"], row[metric + "_sd"]
                xs.append(x); ys.append(y)
                values = observed(method, architecture, fraction)
                ax.scatter([value[xmetric] for value in values], [value[metric] for value in values],
                           color=COLORS[index], marker=MARKERS[index], s=30, alpha=0.4, linewidths=0.7, zorder=2)
                ax.errorbar(x, y, yerr=error, color=COLORS[index], marker=MARKERS[index],
                            markersize=7 if index == 0 else 5, capsize=3, linestyle="none", zorder=3)
            ax.plot(xs, ys, color=COLORS[index], linestyle=LINES[index], linewidth=1.25)
        ax.set_title(method)

    try:
        with plt.rc_context(style):
            # Cell means use a shared color scale; seed-level values remain visible in each cell.
            fig, axes = panel_grid(height=6.7, width=max(6.0, len(fractions) * 1.45))
            values = [row["best_val_rmse_km_mean"] for row in data.summaries if row["n_seeds"]]
            low, high = (min(values), max(values)) if values else (0.0, 1.0)
            if low == high:
                high = low + max(abs(low) * 0.1, 1e-6)
            cmap = matplotlib.colormaps["cividis"].with_extremes(bad="#e0e0e0")
            for method, ax in zip(methods, axes[0], strict=True):
                matrix = np.full((len(ARCHITECTURES), len(fractions)), np.nan)
                for y, architecture in enumerate(ARCHITECTURES):
                    for x, fraction in enumerate(fractions):
                        row = summary[(method, architecture, fraction)]
                        if row["n_seeds"]:
                            matrix[y, x] = row["best_val_rmse_km_mean"]
                im = ax.imshow(matrix, cmap=cmap, vmin=low, vmax=high, aspect="auto", interpolation="nearest")
                for y, architecture in enumerate(ARCHITECTURES):
                    for x, fraction in enumerate(fractions):
                        row = summary[(method, architecture, fraction)]
                        if not row["n_seeds"]:
                            label = "NOT_RUN"
                        else:
                            error = row["best_val_rmse_km_sd"]
                            spread = f" ± {error:.3f}" if error is not None else " (SD unavailable)"
                            seed_values = ", ".join(f"{item['best_val_rmse_km']:.3f}" for item in observed(method, architecture, fraction))
                            label = f"{row['best_val_rmse_km_mean']:.3f}{spread}\n[{seed_values}]\nn={row['n_seeds']}/{len(seeds)}"
                        color = "white" if row["n_seeds"] and matrix[y, x] < low + (high-low)*0.43 else "#111111"
                        ax.text(x, y, label, ha="center", va="center", fontsize=8, color=color)
                labels = []
                for fraction in fractions:
                    counts = {summary[(method, architecture, fraction)]["train_count"] for architecture in ARCHITECTURES} - {None}
                    labels.append((f"N={next(iter(counts))}" if len(counts) == 1 else "N=NOT_RUN") + f"\nf={fraction:g}")
                ax.set(xticks=range(len(fractions)), xticklabels=labels, yticks=range(len(ARCHITECTURES)),
                       yticklabels=[LABELS[key] for key in ARCHITECTURES], title=method, xlabel="Actual training count / planned fraction")
                ax.grid(False)
            fig.subplots_adjust(left=0.25 if len(methods) == 1 else 0.12, right=0.87, bottom=0.17)
            color_ax = fig.add_axes((0.89, 0.25, 0.015, 0.55))
            fig.colorbar(im, cax=color_ax, label="Best validation RMSE (km)")
            finish(fig, "validation-rmse-heatmap", "Training count × network scale: validation RMSE",
                   "Cell means and sample SD share one sequential color scale; brackets contain individual seed values in planned seed order. Gray cells are NOT_RUN. Baseline and medium are distinct architectures.", legend=False)

            fig, axes = panel_grid()
            for method, ax in zip(methods, axes[0], strict=True):
                sample_curve(ax, method, "test_rmse_km")
                ax.set(xlabel="Actual training samples N", ylabel="Held-out test RMSE (km)")
            finish(fig, "sample-count-test-rmse", "Actual training count and held-out test performance",
                   "Separate curves preserve all six choices. Small translucent points are individual seeds and larger markers carry arithmetic mean ± sample SD. Missing fractions break curves; auto/fixed duplicate profiles are not pooled.")

            fig, axes = panel_grid(rows=len(fractions), columns=len(methods), height=3.1)
            for row_index, fraction in enumerate(fractions):
                for method, ax in zip(methods, axes[row_index], strict=True):
                    for index, architecture in enumerate(ARCHITECTURES):
                        group = summary[(method, architecture, fraction)]
                        if not group["n_seeds"]:
                            continue
                        points = observed(method, architecture, fraction)
                        ax.scatter([point["parameter_count"] for point in points], [point["test_rmse_km"] for point in points], color=COLORS[index], marker=MARKERS[index], alpha=0.4, s=30)
                        ax.errorbar(group["parameter_count_mean"], group["test_rmse_km_mean"], yerr=group["test_rmse_km_sd"], color=COLORS[index], marker=MARKERS[index], linestyle="none", capsize=3)
                    known = [summary[(method, architecture, fraction)]["train_count"] for architecture in ARCHITECTURES if summary[(method, architecture, fraction)]["n_seeds"]]
                    ax.set(title=f"{method} | N={known[0] if known else 'NOT_RUN'} | f={fraction:g}", xlabel="Trainable parameters", ylabel="Test RMSE (km)")
                    ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
            finish(fig, "parameter-count-test-rmse", "Parameter count and test error at each data budget",
                   "Each row is a planned data fraction, labeled with realized N. Positions encode actual parameter counts and error. Coincident auto/fixed points represent the same profile, not independent architecture evidence.")

            fig, axes = panel_grid(rows=len(fractions), columns=len(methods), height=3.1)
            for row_index, fraction in enumerate(fractions):
                for method, ax in zip(methods, axes[row_index], strict=True):
                    for index, architecture in enumerate(ARCHITECTURES):
                        group = summary[(method, architecture, fraction)]
                        if not group["n_seeds"]:
                            continue
                        points = observed(method, architecture, fraction)
                        ax.scatter([point["total_train_wall_s"] for point in points], [point["test_rmse_km"] for point in points], color=COLORS[index], marker=MARKERS[index], alpha=0.4, s=30)
                        ax.errorbar(group["total_train_wall_s_mean"], group["test_rmse_km_mean"],
                                    xerr=group["total_train_wall_s_sd"], yerr=group["test_rmse_km_sd"],
                                    color=COLORS[index], marker=MARKERS[index], linestyle="none", capsize=3)
                    ax.set(title=f"{method} | f={fraction:g}", xlabel="Observed training wall time (s)", ylabel="Test RMSE (km)")
            finish(fig, "training-cost-test-rmse", "Observed training cost and held-out quality",
                   "Raw seed points and horizontal/vertical sample SD describe runtime and test RMSE for each data budget. Times include the training command and are not controlled performance benchmarks.")

            for method in methods:
                fig, axes = panel_grid(rows=2, columns=len(fractions), height=3.4, width=3.8)
                for column, fraction in enumerate(fractions):
                    for row_index, split in enumerate(("train", "val")):
                        ax = axes[row_index, column]
                        for index, architecture in enumerate(ARCHITECTURES):
                            selected = [row for row in data.histories if row["method"] == method and row["architecture"] == architecture and row["train_fraction"] == fraction]
                            epochs = sorted({row["epoch"] for row in selected})
                            means, deviations = [], []
                            for epoch in epochs:
                                values = [row[f"{split}_rmse_km"] for row in selected if row["epoch"] == epoch]
                                means.append(statistics.mean(values))
                                deviations.append(statistics.stdev(values) if len(values) > 1 else 0.0)
                            for seed in seeds:
                                seed_rows = sorted((row for row in selected if row["seed"] == seed), key=lambda row: row["epoch"])
                                ax.plot([row["epoch"] for row in seed_rows], [row[f"{split}_rmse_km"] for row in seed_rows], color=COLORS[index], alpha=0.2, linewidth=0.7)
                            if epochs:
                                ax.plot(epochs, means, color=COLORS[index], linestyle=LINES[index], marker=MARKERS[index], markevery=max(1, len(epochs)//8), markersize=3, linewidth=1.25)
                                if len({row["seed"] for row in selected}) > 1:
                                    ax.fill_between(epochs, np.asarray(means)-deviations, np.asarray(means)+deviations, color=COLORS[index], alpha=0.1)
                        ax.set(title=f"f={fraction:g} | {'online training' if split == 'train' else 'validation'}", xlabel="Epoch", ylabel="RMSE (km)")
                        if not any(row["method"] == method and row["train_fraction"] == fraction for row in data.histories):
                            ax.text(0.5, 0.5, "NOT_RUN", transform=ax.transAxes, ha="center")
                finish(fig, f"learning-curves-{method}", f"{method}: online training and validation trajectories",
                       "Thin lines are individual seeds; bold lines and bands are across-seed mean ± sample SD at each epoch. Online training uses training-mode dropout/BN while weights update; it must not be interpreted as a frozen-model training score or a clean generalization gap.")

            fig, axes = panel_grid(rows=len(methods), columns=2, width=5.8)
            chosen_handles = [Line2D([], [], color=color, marker=marker, linestyle=line, label=label)
                              for color, marker, line, label in ((COLORS[1], "o", "-", "Fixed profile selected by validation"),
                                                               (COLORS[0], "*", "--", "Current uncalibrated auto rule"),
                                                               (COLORS[5], "X", ":", "Original baseline reference"))]
            for method, (capacity_ax, error_ax) in zip(methods, axes, strict=True):
                for prefix, color, marker, line in (("selected", COLORS[1], "o", "-"), ("auto", COLORS[0], "*", "--"), ("baseline", COLORS[5], "X", ":")):
                    xs, errors, capacities = [], [], []
                    for fraction_index, fraction in enumerate(fractions):
                        cell = next(row for row in selection if row["method"] == method and row["train_fraction"] == fraction)
                        if cell["status"] != "PASS" or (prefix != "selected" and cell[prefix + "_status"] != "PASS"):
                            xs.append(float("nan")); errors.append(float("nan")); capacities.append(float("nan")); continue
                        x, y = cell["train_count"], cell[prefix + "_test_rmse_km_mean"]
                        arch = cell["selected_architecture"] if prefix == "selected" else "complex_CNN" if prefix == "auto" else "complex_CNN_baseline"
                        group = summary[(method, arch, fraction)]
                        capacity = group["parameter_count_mean"]
                        xs.append(x); errors.append(y); capacities.append(capacity)
                        points = observed(method, arch, fraction)
                        for ax, metric, mean, deviation in (
                            (capacity_ax, "parameter_count", capacity, group["parameter_count_sd"]),
                            (error_ax, "test_rmse_km", y, cell[prefix + "_test_rmse_km_sd"]),
                        ):
                            ax.scatter([point["train_count"] for point in points], [point[metric] for point in points], color=color, marker=marker, s=30, alpha=0.4)
                            ax.errorbar(x, mean, yerr=deviation, color=color, marker=marker, linestyle="none", capsize=3)
                        offset = {"selected": 12, "auto": -18, "baseline": 29}[prefix]
                        alignment = "left" if fraction_index == 0 else "right" if fraction_index == len(fractions)-1 else "center"
                        if prefix != "baseline":
                            capacity_ax.annotate(" / ".join(group["selected_profiles"]), (x, capacity), xytext=(0, offset),
                                                 textcoords="offset points", ha=alignment, color=color, fontsize=8)
                    capacity_ax.plot(xs, capacities, color=color, linestyle=line)
                    error_ax.plot(xs, errors, color=color, linestyle=line)
                capacity_ax.set(title=f"{method}: selected capacity", xlabel="Actual training samples N", ylabel="Trainable parameters")
                capacity_ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
                capacity_ax.margins(x=0.1, y=0.2)
                error_ax.set(title=f"{method}: held-out performance", xlabel="Actual training samples N", ylabel="Test RMSE (km)")
            finish(fig, "validation-selected-reference", "Training count, selected capacity, and held-out performance",
                   "Left: actual parameter counts with profile labels; right: test RMSE after the choice. Both panels show raw seeds and mean ± sample SD (coincident parameter counts may have zero spread). At each N, the reference profile is selected among small/medium/large/xlarge using mean best validation RMSE, with exact ties favoring fewer parameters. This is development-validation selection, not a newly calibrated rule or independent validation of one. Auto/fixed matches remain separate series, not additional replicates.", selected_handles=chosen_handles)
    finally:
        plt.close("all")

    _csv(output_dir / "metrics.csv", data.rows)
    _csv(output_dir / "summary.csv", data.summaries)
    _csv(output_dir / "missing.csv", [row for row in data.rows if row["status"] != "PASS"])
    _csv(output_dir / "histories.csv", data.histories)
    _csv(output_dir / "validation-selection.csv", selection)
    _json(output_dir / "figure-captions.json", captions)
    _json(output_dir / "study-results.json", {"schema_version": 1, "pack": data.study["pack"], "study_id": data.study["study_id"],
                                               "status": data.status, "study": data.study, "metrics": data.rows,
                                               "summaries": data.summaries, "validation_selection": selection,
                                               "inventories": data.inventories, "limitations": LIMITATIONS})
    manifest = {"schema_version": 1, "pack": data.study["pack"], "study_id": data.study["study_id"], "status": data.status,
                "created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": data.study["source_sha256"],
                "plan_sha256": data.study["plan_sha256"], "formats": ["pdf", "png"], "preview_dpi": dpi,
                "inputs": data.sources, "renderer_sources": renderer_sources, "renderer_provenance": renderer_provenance, "figures": figures,
                "dependencies": {name: importlib.metadata.version(name) for name in ("matplotlib", "numpy")},
                "aggregation": {"replicate": "planned seed", "mean": "arithmetic", "spread": "sample SD ddof=1, unavailable for n<2",
                                "missing": "explicit NOT_RUN; no imputation or connecting missing fractions",
                                "selection": "fixed scalable profile by mean best validation RMSE; no test-based selection",
                                "automatic_duplicates": "separate series, never additional replicates"},
                "limitations": LIMITATIONS,
                "files": [{"path": path.relative_to(output_dir).as_posix(), "size": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in sorted(output_dir.iterdir()) if path.is_file()]}
    _json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, help="Root containing complete <run_id> evidence directories; defaults to the study directory's parent")
    parser.add_argument("--output-dir", type=Path, required=True, help="New destination for PDF/PNG figures and source tables")
    parser.add_argument("--dpi", type=int, default=160, help="PNG preview DPI")
    parser.add_argument("--allow-renderer-update", action="store_true",
                        help="Explicitly allow renderer source changes; record hashes and redraw without retraining, while preserving all evidence checks")
    args = parser.parse_args(argv)
    try:
        result = render_study(args.study, args.output_dir, runs_root=args.runs_root, dpi=args.dpi,
                              allow_renderer_update=args.allow_renderer_update)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Study visualization failed: {exc}\n")
    print(f"{result['status']}: {len(result['figures'])} PDF/PNG files in {args.output_dir}")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
