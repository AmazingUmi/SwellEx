#!/usr/bin/env python3
"""Plot a single SwellEx run. Execute with the frozen uv environment on haiqin."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.data_scale_pack.visualization.results import ModelResult, load_run
from scripts.data_scale_pack.localization_networks.common.prediction_utils import save_prediction_plot
from scripts.data_scale_pack.localization_networks.workflow.tasks import ARCHITECTURES, METHODS, model_plan

COLORS = ("#0072B2", "#9A5300", "#007A5E", "#805080", "#444444", "#A03623")
MARKERS = ("o", "s", "^", "D", "v", "P")


def prediction_grid_layout(models: list[ModelResult]) -> list[list[int | None]]:
    """Map original model indices to feature rows and shared architecture columns.

    Rows/columns absent from the selected models are omitted. A missing model
    within a retained feature/architecture pair leaves an empty panel.
    """
    if not models:
        raise ValueError("Prediction grid requires at least one model")
    canonical = {(item["method"], item["model"]): item["architecture"]
                 for item in model_plan("experiment")}
    identities = [(model.method, model.model) for model in models]
    if len(identities) != len(set(identities)) or any(key not in canonical for key in identities):
        raise ValueError("Prediction grid requires unique canonical method/model pairs")
    selected_methods = {method for method, _ in identities}
    selected_architectures = {canonical[key] for key in identities}
    methods = [method for method in METHODS if method in selected_methods]
    architectures = [architecture for architecture in ARCHITECTURES if architecture in selected_architectures]
    grid = [[None for _ in architectures] for _ in methods]
    for index, identity in enumerate(identities):
        grid[methods.index(identity[0])][architectures.index(canonical[identity])] = index
    return grid


def render_run(run_dir: Path, output_dir: Path, dpi=300,
               models: list[str] | None = None, *, require_pass=True) -> dict:
    """Render verified CSV/JSON results; never overwrite an existing destination.

    The internal remote render driver uses require_pass=False after all model
    evaluations have passed. The public CLI requires the final PASS report.
    """
    if type(dpi) is not int or not 72 <= dpi <= 1200:
        raise ValueError("dpi must be an integer between 72 and 1200")
    data = load_run(Path(run_dir), models, require_pass=require_pass)
    output_dir = Path(output_dir)
    # The source report captured below must not replace any input result.
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"Output already exists: {output_dir}; select a new directory")
    for source in data.sources:
        if (data.run_dir/source["path"]).resolve().is_relative_to(output_dir.resolve()):
            raise ValueError("Output directory may not contain source results")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir/"source-report.json").write_bytes(data.report_bytes)
    figures = []
    context = f"{data.report['run_id']} | {data.report['suite']} | {data.report['purpose']}"
    test_note = f"n={len(data.models[0].predictions)} valid test windows; all windows used; scaling policy uncalibrated"

    def save(fig, stem: str, title: str, note: str = test_note):
        width = max(40, int(fig.get_figwidth()*10))
        lines = [textwrap.fill(line, width=width, break_long_words=False) for line in (title+"\n"+context+"\n"+note).splitlines()]
        fig.suptitle("\n".join(lines), fontsize=11)
        path = output_dir/f"{stem}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="pdf", dpi=dpi, facecolor="white")
        figures.append(path.relative_to(output_dir).as_posix())
        plt.close(fig)

    def axes_grid():
        grid = prediction_grid_layout(data.models)
        rows, columns = len(grid), len(grid[0])
        fig, axes = plt.subplots(rows, columns, figsize=(4.5*columns, 3.5*rows+0.9),
                                 squeeze=False, sharex=True, sharey=True, layout="constrained")
        model_axes = [None] * len(data.models)
        for cells, axis_row in zip(grid, axes):
            for index, ax in zip(cells, axis_row):
                if index is None:
                    ax.set_visible(False)
                else:
                    ax.tick_params(labelbottom=True, labelleft=True)
                    model_axes[index] = ax
        return fig, model_axes

    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True,
             "grid.alpha": 0.2, "axes.spines.top": False, "axes.spines.right": False,
             "pdf.fonttype": 42, "savefig.transparent": False,
             "legend.edgecolor": "#b0b0b0", "legend.framealpha": 1.0, "legend.fancybox": False}
    try:
        with plt.rc_context(style):
            fig, axes = axes_grid()
            for index, (model, ax) in enumerate(zip(data.models, axes)):
                rows = model.predictions
                ax.scatter([r.time_s for r in rows], [r.true_km for r in rows], marker="x",
                           color="#222222", s=26, label="Label")
                ax.scatter([r.time_s for r in rows], [r.predicted_km for r in rows],
                           marker=MARKERS[index % len(MARKERS)], color=COLORS[index % len(COLORS)],
                           s=22, alpha=0.85, label="Prediction")
                ax.set(title=model.caption, xlabel="Window center time (s)", ylabel="Range (km)")
                ax.legend(fontsize=8)
            save(fig, "range-time", "Source range predictions")

            fig, axes = axes_grid()
            values = [value for model in data.models for r in model.predictions
                      for value in (r.true_km, r.predicted_km)]
            low, high = min(values), max(values)
            padding = max((high-low)*0.06, 1e-4)
            low, high = low-padding, high+padding
            for index, (model, ax) in enumerate(zip(data.models, axes)):
                ax.plot([low, high], [low, high], "--", color="#555555", lw=1, label="Ideal: y=x")
                ax.scatter([r.true_km for r in model.predictions], [r.predicted_km for r in model.predictions],
                           color=COLORS[index % len(COLORS)], marker=MARKERS[index % len(MARKERS)], s=25)
                ax.set(xlim=(low, high), ylim=(low, high), title=model.caption,
                       xlabel="Label range (km)", ylabel="Predicted range (km)")
                ax.set_aspect("equal", adjustable="box")
                ax.legend(fontsize=8)
            save(fig, "prediction-scatter", "Predictions versus labels")

            fig, axes = axes_grid()
            bound = max(abs(r.residual_km) for model in data.models for r in model.predictions)
            bound = max(bound*1.08, 1e-6)
            for index, (model, ax) in enumerate(zip(data.models, axes)):
                ax.axhline(0, color="#555555", ls="--", lw=1)
                ax.scatter([r.time_s for r in model.predictions], [r.residual_km for r in model.predictions],
                           color=COLORS[index % len(COLORS)], marker=MARKERS[index % len(MARKERS)], s=25)
                ax.set(ylim=(-bound, bound), title=model.caption, xlabel="Window center time (s)",
                       ylabel="Prediction − label (km)")
            save(fig, "residuals-time", "Signed prediction errors")

            fig, ax = plt.subplots(figsize=(12, max(5.5, len(data.models)*0.25)), layout="constrained")
            for index, model in enumerate(data.models):
                errors = sorted(abs(r.residual_km) for r in model.predictions)
                proportions = [(i+1)/len(errors) for i in range(len(errors))]
                ax.step([0, *errors], [0, *proportions], where="post",
                        color=COLORS[index % len(COLORS)], label=model.label,
                        marker=MARKERS[index % len(MARKERS)], markersize=3,
                        linestyle=("-", "--", ":")[index//len(COLORS) % 3])
            ax.set(xlabel="Absolute range error (km)", ylabel="Fraction of observed test windows",
                   xlim=(0, None), ylim=(0, 1.02))
            ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, edgecolor="#b0b0b0")
            save(fig, "absolute-error-ecdf", "Empirical absolute-error distribution")

            fig, axes = plt.subplots(1, 3, figsize=(12, max(3.5, len(data.models)*0.5+1.4)),
                                      sharey=True, layout="constrained")
            positions = list(range(len(data.models)))
            for ax, key, title in zip(axes, ("mae_km", "rmse_km", "max_error_km"), ("MAE", "RMSE", "Maximum absolute error")):
                ax.barh(positions, [m.metrics[key] for m in data.models],
                        color=[COLORS[i % len(COLORS)] for i in positions])
                ax.set(xlim=(0, None), xlabel="Range error (km)", title=title)
                ax.grid(axis="y", visible=False)
            axes[0].set_yticks(positions, [m.label for m in data.models], fontsize=9)
            axes[0].invert_yaxis()
            save(fig, "error-metrics", "Test-window error summaries (single run; no uncertainty estimate)")

            for model in data.models:
                prediction_path = output_dir/f"models/{model.method}/{model.model}/test_outputs/prediction.pdf"
                rows = [(row.sample, row.source_index, row.time_s, row.predicted_km,
                         row.true_km, abs(row.residual_km)) for row in model.predictions]
                save_prediction_plot(rows, prediction_path,
                                     title=f"{model.caption}\n{data.report['run_id']}")
                figures.append(prediction_path.relative_to(output_dir).as_posix())
                if model.history:
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
                    epochs = [r["epoch"] for r in model.history]
                    mode = model.history[0]["loss_space"]
                    for ax, key, label in zip(axes, ("loss", "mae_km", "rmse_km"),
                                               (f"SmoothL1 loss ({mode} space)", "MAE (km)", "RMSE (km)")):
                        for split, color, marker, linestyle in (("train", COLORS[0], "o", "-"), ("val", COLORS[1], "s", "--")):
                            ax.plot(epochs, [r[f"{split}_{key}"] for r in model.history], label=split,
                                    color=color, marker=marker, ls=linestyle)
                        ax.set(xlabel="Epoch", ylabel=label, ylim=(0, None))
                        if len(epochs) <= 12:
                            ax.set_xticks(epochs)
                        ax.legend(fontsize=8)
                    save(fig, f"models/{model.method}/{model.model}/training/history", model.caption+" training history",
                         f"{len(epochs)} recorded epochs; original train/validation metrics")
    finally:
        plt.close("all")

    columns = ("pack", "run_id", "split_strategy", "train_side", "seed", "train_fraction",
               "method", "model", "selection_mode", "selected_profile", "train_count",
               "parameter_count", "parameter_budget", "scaling_policy_version", "calibration_status",
               "base_channels", "stage_repeats", "stage_strides", "samples", "mae_km", "rmse_km", "bias_km", "max_error_km",
               "max_error_window_center_s", "first_window_center_s", "last_window_center_s")
    with (output_dir/"metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model in data.models:
            row = {**model.metrics, **model.architecture["scaling"],
                   "pack": data.report["pack"], "run_id": data.report["run_id"],
                   "method": model.method, "model": model.model,
                   "split_strategy": data.config["dataset"]["split_strategy"],
                   "train_side": data.config["dataset"]["train_side"],
                   "seed": data.config["validation"]["seed"],
                   "train_fraction": model.architecture["sampling"]["train_fraction"],
                   "base_channels": model.architecture["model_config"]["base_channels"]}
            for field in ("stage_repeats", "stage_strides"):
                row[field] = json.dumps(row[field], separators=(",", ":"))
            writer.writerow({key: row[key] for key in columns})
    manifest = {"schema_version": 1, "status": "PASS", "pack": data.report["pack"], "run_id": data.report["run_id"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "suite": data.report["suite"], "purpose": data.report["purpose"],
                "configuration": data.config, "config_sha256": data.report["config_sha256"],
                "model_plan": data.report["model_plan"], "selection": data.report.get("selection"),
                "source_sha256": data.report.get("source_sha256"), "uv_lock_sha256": data.report.get("uv_lock_sha256"),
                "run_dir": str(data.run_dir), "formats": ["pdf"], "dpi": dpi,
                "render_output_dir": str(output_dir.resolve()),
                "dependencies": {name: importlib.metadata.version(name) for name in ("matplotlib", "numpy")},
                "inputs": data.sources, "captured_report": "source-report.json", "figures": figures,
                "models": [{**m.architecture, **m.metrics} for m in data.models],
                "transformations": {"range_units": "km", "time_units": "s",
                                    "residual": "prediction minus label",
                                    "metrics": "recomputed from every CSV row, without pooling runs",
                                    "ecdf": "sorted absolute errors; rank/n; no bins or smoothing",
                                    "missing_windows": "CSV contains valid windows only; no interpolation or imputation",
                                    "training": "original epoch metrics; loss space stated separately for each model",
                                    "scaling": "frozen training-count architecture selection; policy uncalibrated; no test-driven selection"}}
    manifest["renderer_sources"] = [
        {"path": path.relative_to(ROOT).as_posix(),
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (Path(__file__).resolve(), ROOT/"scripts/data_scale_pack/visualization/results.py",
                     ROOT/"scripts/data_scale_pack/localization_networks/common/prediction_utils.py",
                     ROOT/"scripts/data_scale_pack/localization_networks/workflow/tasks.py",
                     ROOT/"scripts/run_identity.py", ROOT/"scripts/packs.py")
    ]
    manifest["delivery_roots"] = {"figure": f"output/figure/{data.report['run_id']}",
                                  "data": f"output/data/{data.report['run_id']}/reports/visualization"}
    manifest["files"] = [{"path": p.relative_to(output_dir).as_posix(), "size": p.stat().st_size,
                          "group": "figure" if p.suffix == ".pdf" else "data",
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in sorted(output_dir.rglob("*")) if p.is_file()]
    (output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run", required=True, type=Path, help="One run directory containing report.json and model CSV/JSON")
    parser.add_argument("--output-dir", type=Path, help="New directory; default: output/figure/<run_id>")
    parser.add_argument("--model", action="append", help="METHOD/MODEL from the run report; repeat to select models")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    output = args.output_dir if args.output_dir is not None else ROOT/"output/figure"/args.run.resolve().name
    try:
        result = render_run(args.run, output, dpi=args.dpi, models=args.model)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Visualization failed: {exc}\n")
    print(f"PASS: {len(result['figures'])} figures in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
