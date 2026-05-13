"""Training loop and metric helpers for SCM range regression."""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from common.h5_utils import resolve_h5_paths
from common.training_utils import (
    resolve_resume_checkpoint,
    run_epoch,
    seed_everything,
    target_stats,
)

from .data import fixed_split_bundle, random_split_bundle
from .model import (
    build_model,
    build_model_config,
    model_config_from_checkpoint,
    model_uses_dataset_input_norm,
    serialize_model_config,
)
from .paths import (
    dataset_train_glob,
    dataset_val_glob,
    safe_name,
    time_suffix,
    train_output_dir,
    with_time_suffix,
)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    resume_path = resolve_resume_checkpoint(args)
    resume_checkpoint = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=args.device)

    if args.train_data is None:
        train_paths = resolve_h5_paths([dataset_train_glob(args.data)])
        split_mode = "dataset_train_random_split"
    else:
        train_paths = resolve_h5_paths(args.train_data)
        split_mode = "explicit_train_data_random_split"

    if args.val_data is None:
        val_paths = resolve_h5_paths([dataset_val_glob(args.data)], required=False)
        if not val_paths:
            val_paths = None
    else:
        val_paths = resolve_h5_paths(args.val_data)

    source_paths = train_paths if not val_paths else [*train_paths, *val_paths]
    if val_paths:
        split_mode = "dataset_explicit_train_val_split"
        if args.train_data is not None or args.val_data is not None:
            split_mode = "explicit_train_val_split"

    if resume_checkpoint is None:
        input_normalized = (not args.no_input_norm) and model_uses_dataset_input_norm(
            args.model
        )
    else:
        input_normalized = bool(resume_checkpoint.get("input_normalized_per_sample", True))
    if val_paths is None:
        data_bundle = random_split_bundle(
            train_paths,
            normalize_input=input_normalized,
            val_fraction=args.val_fraction,
            seed=args.seed,
            split_mode=split_mode,
        )
    else:
        data_bundle = fixed_split_bundle(
            train_paths,
            val_paths,
            normalize_input=input_normalized,
            split_mode=split_mode,
        )
    input_shape = data_bundle.input_shape
    train_count = len(data_bundle.train_idx)
    val_count = len(data_bundle.val_idx)

    if resume_checkpoint is None:
        y_mean, y_std = target_stats(data_bundle.train_labels)
        loss_space = args.loss_space
    else:
        y_mean = float(resume_checkpoint["target_mean_km"])
        y_std = float(resume_checkpoint["target_std_km"])
        loss_space = str(resume_checkpoint.get("loss_space", args.loss_space))
        if loss_space != args.loss_space:
            print(
                "Resume checkpoint loss space overrides CLI: "
                f"{loss_space} (CLI requested {args.loss_space})"
            )

    train_loader = DataLoader(
        data_bundle.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        data_bundle.val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    _, pairs, freq_bins = input_shape
    if resume_checkpoint is None:
        model_name = args.model
        config = build_model_config(
            model_name,
            input_shape,
            base_channels=args.base_channels,
            dropout=args.dropout,
        )
    else:
        model_name, config = model_config_from_checkpoint(resume_checkpoint)
        if config.input_pairs != pairs or config.input_freq_bins != freq_bins:
            raise ValueError(
                "Resume checkpoint input shape does not match current data: "
                f"checkpoint has {config.input_pairs} pairs x "
                f"{config.input_freq_bins} freq bins, "
                f"data has {pairs} pairs x {freq_bins} freq bins."
            )
    device = torch.device(args.device)
    model = build_model(model_name, config).to(device)
    criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    start_epoch = 1
    end_epoch = args.epochs
    best_rmse = float("inf")
    history: list[dict[str, float]] = []

    if resume_checkpoint is not None:
        checkpoint_dataset = resume_checkpoint.get("dataset_name")
        if checkpoint_dataset is not None and checkpoint_dataset != args.data:
            raise ValueError(
                f"Resume checkpoint dataset '{checkpoint_dataset}' does not match "
                f"--data '{args.data}'."
            )
        model.load_state_dict(resume_checkpoint["model_state"])
        if "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])

        completed_epoch = int(resume_checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        end_epoch = completed_epoch + args.epochs
        best_rmse = float(
            resume_checkpoint.get(
                "best_rmse_km",
                resume_checkpoint.get("metrics", {}).get("val_rmse_km", float("inf")),
            )
        )
        history = list(resume_checkpoint.get("history", []))

    run_suffix = time_suffix()
    dataset_name = safe_name(args.data)
    output_dir = train_output_dir(args.output_dir, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = with_time_suffix(output_dir / f"{dataset_name}_best.pt", run_suffix)
    last_path = with_time_suffix(output_dir / f"{dataset_name}_last.pt", run_suffix)
    history_path = with_time_suffix(output_dir / f"{dataset_name}_history.json", run_suffix)

    print(f"Split mode: {split_mode}")
    print(f"Source files: {len(source_paths)}")
    print(f"Dataset: {args.data}")
    print(f"Model: {model_name}")
    print(f"Train/val samples: {train_count}/{val_count}")
    print(
        "Input: "
        f"2 x {pairs} pair-vector entries x {freq_bins} freq bins"
    )
    print(f"Target normalization: mean={y_mean:.4f} km, std={y_std:.4f} km")
    print(f"Loss space: {loss_space} (SmoothL1 beta={args.huber_beta:g})")
    print(f"Device: {device}")
    print(f"Run suffix: {run_suffix}")
    if resume_checkpoint is not None:
        print(f"Resume checkpoint: {resume_path}")
        print(f"Resume epochs: {start_epoch}-{end_epoch}")

    for epoch in range(start_epoch, end_epoch + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            y_mean,
            y_std,
            loss_space,
            optimizer,
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, y_mean, y_std, loss_space
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "loss_space": loss_space,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | "
            f"train loss {row['train_loss']:.4f} mae {row['train_mae_km']:.3f} km | "
            f"val loss {row['val_loss']:.4f} mae {row['val_mae_km']:.3f} km "
            f"rmse {row['val_rmse_km']:.3f} km"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "dataset_name": args.data,
            "model_name": model_name,
            "model_config": serialize_model_config(config),
            "target_mean_km": y_mean,
            "target_std_km": y_std,
            "loss_space": loss_space,
            "huber_beta": args.huber_beta,
            "input_normalized_per_sample": input_normalized,
            "input_shape": input_shape,
            "split_mode": split_mode,
            "source_h5_paths": [str(p) for p in source_paths],
            "val_fraction": args.val_fraction,
            "train_indices": data_bundle.train_idx,
            "val_indices": data_bundle.val_idx,
            "epoch": epoch,
            "metrics": row,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_rmse_km": min(best_rmse, val_metrics["rmse_km"]),
            "history": history,
        }
        if val_metrics["rmse_km"] < best_rmse:
            best_rmse = val_metrics["rmse_km"]
            checkpoint["best_rmse_km"] = best_rmse
            torch.save(checkpoint, best_path)
        torch.save(checkpoint, last_path)

        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
