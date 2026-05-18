"""CLI for standalone SCM-GRNN reference building and prediction."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from grnn_data import GrnnFeatureBundle, load_scm_grnn_features
from grnn_model import ScmGrnnRegressor
from grnn_paths import (
    DEFAULT_OUTPUT_DIR,
    dataset_test_glob,
    dataset_train_glob,
    latest_time_suffixed_path,
    reference_output_dir,
    resolve_h5_paths,
    safe_name,
    test_output_dir,
    time_suffix,
    with_time_suffix,
)


PredictionRow = tuple[int, int, float, float, float, float]


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_sigma_candidates() -> list[float]:
    return [
        *[round(0.01 * i, 2) for i in range(1, 11)],
        *[round(0.1 * i, 2) for i in range(2, 21)],
    ]


def resolve_reference_path(args: argparse.Namespace) -> Path:
    if args.reference is not None:
        path = args.reference
        if not path.exists():
            raise FileNotFoundError(f"Reference artifact not found: {path}")
        return path

    dataset_name = safe_name(args.data)
    base_path = reference_output_dir(args.output_dir) / f"{dataset_name}_reference.pt"
    if base_path.exists():
        return base_path
    latest_path = latest_time_suffixed_path(base_path)
    if latest_path is not None:
        print(f"Using latest reference artifact: {latest_path}")
        return latest_path
    expected_pattern = base_path.with_name(f"{base_path.stem}_*{base_path.suffix}")
    raise FileNotFoundError(
        "No GRNN reference artifact found. Expected either "
        f"{base_path} or a timestamped file like {expected_pattern}."
    )


def load_artifact(path: Path, device: torch.device) -> dict[str, object]:
    try:
        artifact = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location=device)
    if not isinstance(artifact, dict):
        raise TypeError(f"GRNN artifact must be a dict, got {type(artifact).__name__}.")
    return artifact


def metric_summary(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    err = y_pred.view(-1) - y_true.view(-1)
    mae = float(err.abs().mean())
    rmse = math.sqrt(float(err.square().mean()))
    mape = float((err.abs() / y_true.view(-1).abs().clamp_min(1.0e-12)).mean() * 100.0)
    return {"mae_km": mae, "rmse_km": rmse, "mape_pct": mape}


def select_sigma_by_validation(
    train_bundle: GrnnFeatureBundle,
    val_bundle: GrnnFeatureBundle,
    *,
    candidates: list[float],
    standardize_input: bool,
    batch_size: int,
    device: torch.device,
) -> tuple[float, list[dict[str, float]]]:
    if not candidates:
        raise ValueError("At least one sigma candidate is required.")
    results: list[dict[str, float]] = []
    for sigma in candidates:
        model = ScmGrnnRegressor(
            spread=sigma,
            standardize_input=standardize_input,
            device=device,
        )
        model.fit_reference(train_bundle.x, train_bundle.y_range_km)
        pred = model.predict(val_bundle.x, batch_size=batch_size)
        metrics = metric_summary(val_bundle.y_range_km, pred)
        row = {"sigma": float(sigma), **metrics}
        results.append(row)
        print(
            f"GRNN sigma {sigma:g} | "
            f"val MAE {metrics['mae_km']:.3f} km | "
            f"RMSE {metrics['rmse_km']:.3f} km | "
            f"MAPE {metrics['mape_pct']:.3f}%"
        )
    best = min(results, key=lambda row: row["mape_pct"])
    return float(best["sigma"]), results


def select_sigma_by_cv(
    bundle: GrnnFeatureBundle,
    *,
    candidates: list[float],
    cv_folds: int,
    seed: int,
    standardize_input: bool,
    batch_size: int,
    device: torch.device,
) -> tuple[float, list[dict[str, float]]]:
    n_samples = bundle.x.size(0)
    if cv_folds < 2:
        raise ValueError("Cross-validation requires --cv-folds >= 2.")
    if cv_folds > n_samples:
        raise ValueError(f"--cv-folds ({cv_folds}) cannot exceed samples ({n_samples}).")

    generator = torch.Generator()
    generator.manual_seed(seed)
    shuffled = torch.randperm(n_samples, generator=generator)
    folds = torch.tensor_split(shuffled, cv_folds)

    results: list[dict[str, float]] = []
    for sigma in candidates:
        fold_metrics: list[dict[str, float]] = []
        for fold_id in range(cv_folds):
            val_idx = folds[fold_id]
            train_idx = torch.cat([folds[i] for i in range(cv_folds) if i != fold_id])

            model = ScmGrnnRegressor(
                spread=sigma,
                standardize_input=standardize_input,
                device=device,
            )
            model.fit_reference(bundle.x[train_idx], bundle.y_range_km[train_idx])
            pred = model.predict(bundle.x[val_idx], batch_size=batch_size)
            fold_metrics.append(metric_summary(bundle.y_range_km[val_idx], pred))

        row = {
            "sigma": float(sigma),
            "mean_mae_km": float(sum(m["mae_km"] for m in fold_metrics) / cv_folds),
            "mean_rmse_km": float(sum(m["rmse_km"] for m in fold_metrics) / cv_folds),
            "mean_mape_pct": float(sum(m["mape_pct"] for m in fold_metrics) / cv_folds),
        }
        results.append(row)
        print(
            f"GRNN sigma {sigma:g} | "
            f"CV MAE {row['mean_mae_km']:.3f} km | "
            f"RMSE {row['mean_rmse_km']:.3f} km | "
            f"MAPE {row['mean_mape_pct']:.3f}%"
        )

    best = min(results, key=lambda row: row["mean_mape_pct"])
    return float(best["sigma"]), results


def write_predictions_csv(rows: list[PredictionRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(
            "sample,source_segment_idx,window_center_s,"
            "pred_range_km,true_range_km,abs_error_km\n"
        )
        for row in rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]:.6f},"
                f"{row[3]:.6f},{row[4]:.6f},{row[5]:.6f}\n"
            )


def save_prediction_plot(rows: list[PredictionRow], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing Python dependency for plotting: {exc.name}. "
            "Install dependencies from scripts_py/requirements.txt."
        ) from exc

    sorted_rows = sorted(rows, key=lambda row: row[2])
    time_s = [row[2] for row in sorted_rows]
    pred_km = [row[3] for row in sorted_rows]
    true_km = [row[4] for row in sorted_rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(time_s, true_km, color="#1f77b4", linewidth=1.8, label="Label")
    ax.scatter(time_s, pred_km, color="#d62728", s=24, alpha=0.8, label="Prediction")
    ax.set_xlabel("Window center time (s)")
    ax.set_ylabel("Range (km)")
    ax.set_title("SCM-GRNN Source Range Prediction")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_command(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    train_paths = resolve_h5_paths(
        args.train_data if args.train_data is not None else [dataset_train_glob(args.data)]
    )
    val_paths = (
        resolve_h5_paths(args.val_data, required=True)
        if args.val_data is not None
        else []
    )

    print(f"Loading reference HDF5 files: {len(train_paths)}")
    train_bundle = load_scm_grnn_features(
        train_paths,
        per_sample_channel_norm=args.per_sample_channel_norm,
        max_samples=args.max_reference_samples,
    )

    sigma = float(args.spread)
    search_results: list[dict[str, float]] = []
    sigma_source = "cli_spread"
    candidates = (
        [float(value) for value in args.sigma_candidates]
        if args.sigma_candidates is not None
        else default_sigma_candidates()
    )

    if val_paths:
        print(f"Loading validation HDF5 files for sigma selection: {len(val_paths)}")
        val_bundle = load_scm_grnn_features(
            val_paths,
            per_sample_channel_norm=args.per_sample_channel_norm,
            max_samples=args.max_validation_samples,
        )
        sigma, search_results = select_sigma_by_validation(
            train_bundle,
            val_bundle,
            candidates=candidates,
            standardize_input=args.standardize_input,
            batch_size=args.batch_size,
            device=device,
        )
        sigma_source = "validation"
    elif args.cv_folds > 1:
        sigma, search_results = select_sigma_by_cv(
            train_bundle,
            candidates=candidates,
            cv_folds=args.cv_folds,
            seed=args.seed,
            standardize_input=args.standardize_input,
            batch_size=args.batch_size,
            device=device,
        )
        sigma_source = "cross_validation"
    elif args.sigma_candidates is not None:
        print("--sigma-candidates were provided but no --val-data or --cv-folds > 1 was set.")
        print(f"Using --spread {sigma:g} without sigma search.")

    model = ScmGrnnRegressor(
        spread=sigma,
        standardize_input=args.standardize_input,
        device=device,
    )
    model.fit_reference(train_bundle.x, train_bundle.y_range_km)

    _, pairs, freq_bins = train_bundle.input_shape
    artifact = {
        "artifact_type": "scm_grnn_reference",
        "method": "scm_grnn_range",
        "dataset_name": args.data,
        "input_shape": train_bundle.input_shape,
        "feature_dim": int(train_bundle.x.size(1)),
        "reference_sample_count": int(train_bundle.x.size(0)),
        "source_h5_paths": [str(path) for path in train_bundle.source_paths],
        "per_sample_channel_norm": bool(args.per_sample_channel_norm),
        "sigma_source": sigma_source,
        "sigma_search_results": search_results,
        **model.to_artifact(),
    }

    run_suffix = time_suffix()
    dataset_name = safe_name(args.data)
    ref_dir = reference_output_dir(args.output_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    reference_path = with_time_suffix(ref_dir / f"{dataset_name}_reference.pt", run_suffix)
    torch.save(artifact, reference_path)

    if search_results:
        search_path = with_time_suffix(ref_dir / f"{dataset_name}_sigma_search.json", run_suffix)
        with search_path.open("w", encoding="utf-8") as f:
            json.dump(search_results, f, indent=2)
        print(f"Wrote sigma search: {search_path}")

    print(f"Dataset: {args.data}")
    print(f"Reference files: {len(train_paths)}")
    print(
        "Input: "
        f"2 x {pairs} pair-vector entries x {freq_bins} freq bins "
        f"= {artifact['feature_dim']} GRNN input nodes"
    )
    print(f"Pattern layer size: {artifact['reference_sample_count']}")
    print("Summation layer: 1 weighted S node + 1 unweighted D node")
    print("Output layer: 1 range value in km")
    print(f"Spread sigma: {sigma:g} ({sigma_source})")
    print(f"Feature standardization: {args.standardize_input}")
    print(f"Per-sample channel normalization: {args.per_sample_channel_norm}")
    print(f"Device: {device}")
    print(f"Wrote reference artifact: {reference_path}")


def predict_command(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    reference_path = resolve_reference_path(args)
    artifact = load_artifact(reference_path, device)
    model = ScmGrnnRegressor.from_artifact(artifact, device=device)

    input_paths = resolve_h5_paths(
        args.input_data if args.input_data is not None else [dataset_test_glob(args.data)]
    )
    bundle = load_scm_grnn_features(
        input_paths,
        per_sample_channel_norm=bool(artifact.get("per_sample_channel_norm", False)),
        max_samples=args.max_predict_samples,
    )

    artifact_shape = tuple(int(value) for value in artifact["input_shape"])
    if tuple(bundle.input_shape) != artifact_shape:
        raise ValueError(
            f"Input shape mismatch: reference has {artifact_shape}, "
            f"prediction data has {bundle.input_shape}."
        )

    pred_km = model.predict(bundle.x, batch_size=args.batch_size).view(-1)
    target_km = bundle.y_range_km.view(-1)
    rows: list[PredictionRow] = []
    total_abs_error = 0.0
    for sample, (pred, target) in enumerate(zip(pred_km.tolist(), target_km.tolist())):
        abs_error = abs(pred - target)
        rows.append(
            (
                sample,
                bundle.source_segment_idx[sample],
                bundle.window_center_s[sample],
                pred,
                target,
                abs_error,
            )
        )
        total_abs_error += abs_error

    run_suffix = time_suffix()
    dataset_name = safe_name(args.data)
    if args.predictions_csv is None:
        csv_base = test_output_dir(args.output_dir) / f"{dataset_name}_predictions.csv"
    else:
        csv_base = args.predictions_csv
    csv_path = with_time_suffix(csv_base, run_suffix)
    write_predictions_csv(rows, csv_path)
    print(f"Wrote predictions: {csv_path}")

    if not args.no_plot:
        if args.plot_path is None:
            plot_base = test_output_dir(args.output_dir) / f"{dataset_name}_range_prediction.png"
        else:
            plot_base = args.plot_path
        plot_path = with_time_suffix(plot_base, run_suffix)
        save_prediction_plot(rows, plot_path)
        print(f"Wrote prediction plot: {plot_path}")

    mae = total_abs_error / max(1, len(rows))
    print(f"Reference artifact: {reference_path}")
    print(f"Predicted {len(rows)} samples | MAE {mae:.3f} km")
    for row in rows[: args.print_first]:
        print(
            f"sample {row[0]:04d} source_segment_idx={row[1]}: "
            f"t={row[2]:.3f} s, pred={row[3]:.3f} km, "
            f"true={row[4]:.3f} km, abs_error={row[5]:.3f} km"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone GRNN for MATLAB-exported SCM HDF5 datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser_ = subparsers.add_parser(
        "build",
        help="Build a GRNN reference artifact from SCM HDF5 training files.",
    )
    build_parser_.add_argument("--data", required=True, help="Dataset code under outputs/Datasets.")
    build_parser_.add_argument("--train-data", nargs="+", default=None)
    build_parser_.add_argument("--val-data", nargs="+", default=None)
    build_parser_.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build_parser_.add_argument("--spread", type=float, default=0.01)
    build_parser_.add_argument("--sigma-candidates", nargs="+", type=float, default=None)
    build_parser_.add_argument("--cv-folds", type=int, default=0)
    build_parser_.add_argument("--seed", type=int, default=2026)
    build_parser_.add_argument("--batch-size", type=int, default=128)
    build_parser_.add_argument("--device", default=default_device())
    build_parser_.add_argument("--standardize-input", action="store_true")
    build_parser_.add_argument("--per-sample-channel-norm", action="store_true")
    build_parser_.add_argument("--max-reference-samples", type=int, default=None)
    build_parser_.add_argument("--max-validation-samples", type=int, default=None)
    build_parser_.set_defaults(func=build_command)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run a saved GRNN reference artifact on SCM HDF5 test files.",
    )
    predict_parser.add_argument("--data", required=True, help="Dataset code under outputs/Datasets.")
    predict_parser.add_argument("--input-data", nargs="+", default=None)
    predict_parser.add_argument("--reference", type=Path, default=None)
    predict_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    predict_parser.add_argument("--batch-size", type=int, default=128)
    predict_parser.add_argument("--device", default=default_device())
    predict_parser.add_argument("--predictions-csv", type=Path, default=None)
    predict_parser.add_argument("--plot-path", type=Path, default=None)
    predict_parser.add_argument("--no-plot", action="store_true")
    predict_parser.add_argument("--print-first", type=int, default=10)
    predict_parser.add_argument("--max-predict-samples", type=int, default=None)
    predict_parser.set_defaults(func=predict_command)

    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        raise SystemExit(2)
    args = parser.parse_args()
    args.func(args)
