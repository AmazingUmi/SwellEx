"""Training loop and metric helpers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.base_pack.localization_networks.common.h5_utils import resolve_h5_paths
from scripts.base_pack.localization_networks.common.training_utils import (
    resolve_resume_checkpoint,
    run_epoch,
    seed_everything,
    target_stats,
)

from .bindings import MethodBindings
from .checkpoints import CHECKPOINT_VERSION, validate_checkpoint
from .input_contract import require_matching_input_signature
from .paths import (
    dataset_train_glob, dataset_val_glob, safe_name, time_suffix,
    train_output_dir, with_time_suffix,
)


def train(args: argparse.Namespace, bindings: MethodBindings) -> None:
    if args.no_resume and args.resume_checkpoint is not None:
        raise ValueError("--no-resume and --resume-checkpoint are mutually exclusive.")
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers nonnegative.")
    if not math.isfinite(args.val_fraction) or not 0 < args.val_fraction < 1:
        raise ValueError("val_fraction must be finite and strictly between 0 and 1.")
    for key in ("lr", "weight_decay", "dropout", "huber_beta"):
        if not math.isfinite(getattr(args, key)):
            raise ValueError(f"{key} must be finite.")
    if args.lr <= 0 or args.weight_decay < 0 or args.huber_beta < 0 or not 0 <= args.dropout < 1 or args.base_channels <= 0:
        raise ValueError("Invalid optimizer, loss, or model settings.")
    seed_everything(args.seed)
    resume_path = resolve_resume_checkpoint(args)
    resume_checkpoint = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=args.device)
        validate_checkpoint(resume_checkpoint, method=bindings.method,
                            model_name=args.model, dataset_name=args.data)
        required = {"optimizer_state", "history", "epoch", "best_rmse_km",
                    "best_checkpoint_path", "train_indices", "val_indices", "split_mode", "seed"}
        if not required.issubset(resume_checkpoint):
            raise ValueError(f"Resume checkpoint missing fields: {sorted(required - resume_checkpoint.keys())}")

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
        input_normalized = (not args.no_input_norm) and bindings.model.model_uses_dataset_input_norm(
            args.model
        )
    else:
        input_normalized = resume_checkpoint["input_normalized_per_sample"]
    if val_paths is None:
        data_bundle = bindings.data.random_split_bundle(
            train_paths,
            normalize_input=input_normalized,
            val_fraction=args.val_fraction,
            seed=args.seed,
            split_mode=split_mode,
        )
    else:
        data_bundle = bindings.data.fixed_split_bundle(
            train_paths,
            val_paths,
            normalize_input=input_normalized,
            split_mode=split_mode,
        )
    input_shape = data_bundle.input_shape
    if resume_checkpoint is not None:
        require_matching_input_signature(resume_checkpoint["input_signature"],
                                         data_bundle.input_signature, "Resume checkpoint/data")
        for key, actual in (("train_indices", data_bundle.train_idx),
                            ("val_indices", data_bundle.val_idx),
                            ("split_mode", data_bundle.split_mode), ("seed", args.seed)):
            if resume_checkpoint[key] != actual:
                raise ValueError(f"Resume data split differs from checkpoint: {key}.")
    train_count = len(data_bundle.train_idx)
    val_count = len(data_bundle.val_idx)
    if not train_count or not val_count:
        raise ValueError("Training and validation splits must both contain valid samples.")

    if resume_checkpoint is None:
        y_mean, y_std = target_stats(data_bundle.train_labels)
        loss_space = args.loss_space
    else:
        y_mean = float(resume_checkpoint["target_mean_km"])
        y_std = float(resume_checkpoint["target_std_km"])
        loss_space = resume_checkpoint["loss_space"]
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

    _, feature_count, freq_bins = input_shape
    if resume_checkpoint is None:
        model_name = args.model
        config = bindings.model.build_model_config(
            model_name,
            input_shape,
            base_channels=args.base_channels,
            dropout=args.dropout,
        )
    else:
        model_name, config = bindings.model.model_config_from_checkpoint(resume_checkpoint)
        bindings.check_config_shape(config, input_shape)
    device = torch.device(args.device)
    model = bindings.model.build_model(model_name, config).to(device)
    huber_beta = args.huber_beta if resume_checkpoint is None else resume_checkpoint["huber_beta"]
    criterion = nn.SmoothL1Loss(beta=huber_beta)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_epoch = 1
    end_epoch = args.epochs
    best_rmse = float("inf")
    history: list[dict[str, float]] = []

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])

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

    # Additional epochs form a fresh cosine stage from requested --lr to zero.
    # Restore optimizer momentum, but never continue the old periodic LR curve.
    for group in optimizer.param_groups:
        group["lr"] = args.lr
        group["initial_lr"] = args.lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler_plan = {"policy": "new_cosine_phase", "start_epoch": start_epoch,
                      "phase_epochs": args.epochs, "initial_lr": float(args.lr), "eta_min": 0.0}
    run_suffix = time_suffix()
    dataset_name = safe_name(args.data)
    output_dir = train_output_dir(args.output_dir, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = with_time_suffix(output_dir / f"{dataset_name}_best.pt", run_suffix)
    last_path = with_time_suffix(output_dir / f"{dataset_name}_last.pt", run_suffix)
    history_path = with_time_suffix(output_dir / f"{dataset_name}_history.json", run_suffix)
    actual_best_path = best_path
    if resume_checkpoint is not None:
        actual_best_path = Path(resume_checkpoint["best_checkpoint_path"])
        if not actual_best_path.is_file():
            raise FileNotFoundError(f"Historical best checkpoint not found: {actual_best_path}")
        best_checkpoint = torch.load(actual_best_path, map_location="cpu")
        validate_checkpoint(best_checkpoint, method=bindings.method,
                            model_name=args.model, dataset_name=args.data)
        require_matching_input_signature(data_bundle.input_signature,
                                         best_checkpoint["input_signature"], "Historical best/data")

    print(f"Split mode: {split_mode}")
    print(f"Source files: {len(source_paths)}")
    print(f"Dataset: {args.data}")
    print(f"Model: {model_name}")
    print(f"Train/val samples: {train_count}/{val_count}")
    print(f"Input: 2 x {feature_count} x {freq_bins} ({bindings.method})")
    print(f"Target normalization: mean={y_mean:.4f} km, std={y_std:.4f} km")
    print(f"Loss space: {loss_space} (SmoothL1 beta={huber_beta:g})")
    print(f"Device: {device}")
    print(f"Run suffix: {run_suffix}")
    if resume_checkpoint is not None:
        print(f"Resume checkpoint: {resume_path}")
        print(f"Resume epochs: {start_epoch}-{end_epoch}")
        print(f"Resume scheduler: new cosine phase from --lr={args.lr:g} to 0; optimizer momentum/history restored.")

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
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": epoch_lr,
            "next_lr": float(optimizer.param_groups[0]["lr"]),
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

        improved = val_metrics["rmse_km"] < best_rmse
        if improved:
            actual_best_path = best_path
        checkpoint = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "method": bindings.method,
            "input_signature": data_bundle.input_signature,
            "input_shape": list(input_shape),
            "seed": args.seed,
            "best_checkpoint_path": str(actual_best_path.resolve()),
            "scheduler_plan": scheduler_plan,
            "model_state": model.state_dict(),
            "dataset_name": args.data,
            "model_name": model_name,
            "model_config": bindings.model.serialize_model_config(config),
            "target_mean_km": y_mean,
            "target_std_km": y_std,
            "loss_space": loss_space,
            "huber_beta": huber_beta,
            "input_normalized_per_sample": input_normalized,
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
        if improved:
            best_rmse = val_metrics["rmse_km"]
            checkpoint["best_rmse_km"] = best_rmse
            torch.save(checkpoint, best_path)
        torch.save(checkpoint, last_path)

        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Best checkpoint: {actual_best_path}")
    print(f"Last checkpoint: {last_path}")
