"""Prediction and plotting utilities for ELM pairwise-ratio models."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common.prediction_utils import (
    PredictionRow,
    save_prediction_plot,
    write_predictions_csv,
)

from .data import ElmRangeH5Dataset, resolve_h5_paths
from .model import build_model, model_config_from_checkpoint
from .paths import (
    dataset_test_glob,
    latest_time_suffixed_path,
    safe_name,
    test_output_dir,
    train_output_dir,
    time_suffix,
    with_time_suffix,
)


@torch.no_grad()
def predict(args: argparse.Namespace) -> None:
    dataset_name = safe_name(args.data)
    default_checkpoint_path = (
        train_output_dir(args.output_dir, args.model) / f"{dataset_name}_best.pt"
    )
    checkpoint_path = args.checkpoint or default_checkpoint_path
    if not checkpoint_path.exists():
        latest_checkpoint = latest_time_suffixed_path(checkpoint_path)
        if latest_checkpoint is not None:
            checkpoint_path = latest_checkpoint
            print(f"Using latest checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        expected_pattern = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_*{checkpoint_path.suffix}"
        )
        raise FileNotFoundError(
            "No checkpoint found. Expected either "
            f"{checkpoint_path} or a matching timestamped file like {expected_pattern}."
        )

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model_name, config = model_config_from_checkpoint(checkpoint)
    if model_name != args.model:
        raise ValueError(
            f"Checkpoint model '{model_name}' does not match --model '{args.model}'."
        )
    checkpoint_dataset = checkpoint.get("dataset_name")
    if checkpoint_dataset is not None and checkpoint_dataset != args.data:
        raise ValueError(
            f"Checkpoint dataset '{checkpoint_dataset}' does not match --data '{args.data}'."
        )
    device = torch.device(args.device)
    model = build_model(model_name, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    h5_paths = resolve_h5_paths([dataset_test_glob(args.data)])
    dataset = ElmRangeH5Dataset(
        h5_paths,
        normalize_input=bool(checkpoint.get("input_normalized_per_sample", True)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    y_mean = float(checkpoint["target_mean_km"])
    y_std = float(checkpoint["target_std_km"])
    rows: list[PredictionRow] = []
    offset = 0
    total_abs_error = 0.0
    run_suffix = time_suffix()

    for x, y_km in loader:
        x = x.to(device, non_blocking=True)
        pred_norm = model(x)
        pred_km = (pred_norm.cpu() * y_std + y_mean).view(-1)
        target_km = y_km.view(-1)

        for i, (pred, target) in enumerate(zip(pred_km.tolist(), target_km.tolist())):
            abs_error = abs(pred - target)
            sample = offset + i
            source_segment_idx = dataset.source_segment_idx[sample]
            window_center_s = dataset.window_center_s[sample]
            rows.append(
                (sample, source_segment_idx, window_center_s, pred, target, abs_error)
            )
            total_abs_error += abs_error
        offset += len(target_km)

    if args.predictions_csv is not None:
        predictions_csv_path = args.predictions_csv
    else:
        predictions_csv_path = (
            test_output_dir(args.output_dir, model_name)
            / f"{dataset_name}_predictions.csv"
        )
    if predictions_csv_path is not None:
        predictions_csv = with_time_suffix(predictions_csv_path, run_suffix)
        write_predictions_csv(rows, predictions_csv)
        print(f"Wrote predictions: {predictions_csv}")

    if not args.no_plot:
        plot_path_base = args.plot_path
        if plot_path_base is None:
            plot_path_base = (
                test_output_dir(args.output_dir, model_name)
                / f"{dataset_name}_range_prediction.png"
            )
        plot_path = with_time_suffix(plot_path_base, run_suffix)
        save_prediction_plot(rows, plot_path, title="ELM Source Range Prediction")
        print(f"Wrote prediction plot: {plot_path}")

    mae = total_abs_error / max(1, len(rows))
    print(f"Predicted {len(rows)} samples | MAE {mae:.3f} km")
    for sample, source_segment_idx, window_center_s, pred, target, abs_error in rows[
        : args.print_first
    ]:
        print(
            f"sample {sample:04d} source_segment_idx={source_segment_idx}: "
            f"t={window_center_s:.3f} s, pred={pred:.3f} km, "
            f"true={target:.3f} km, abs_error={abs_error:.3f} km"
        )
