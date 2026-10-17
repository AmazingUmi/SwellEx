"""Training loop and metric helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from scripts.MoE_data_scale.localization_networks.common.h5_utils import resolve_h5_paths
from scripts.MoE_data_scale.localization_networks.common.training_utils import (
    ControlledBatchStream,
    IndexedTrainingDataset,
    controlled_budget_plan,
    controlled_sampler_plan,
    model_state_digest,
    resolve_resume_checkpoint,
    run_epoch,
    seed_everything,
    target_stats,
)

from .bindings import MethodBindings
from .checkpoints import CHECKPOINT_VERSION, validate_checkpoint
from .input_contract import require_matching_input_signature
from .data_selection import select_training_subset, source_identity
from .paths import (
    dataset_train_glob, dataset_val_glob, safe_name, time_suffix,
    train_output_dir, with_time_suffix,
)


def split_inventory(rows) -> dict:
    """Full-precision physical membership, independent of optimizer row order."""
    rows = sorted([(int(source), float(moment), float(label)) for source, moment, label in rows])
    if not rows or len({row[0] for row in rows}) != len(rows) or any(row[0] <= 0 for row in rows):
        raise ValueError("Split inventory requires nonempty unique positive source identities")
    if any(not math.isfinite(value) for row in rows for value in row[1:]):
        raise ValueError("Split inventory requires finite times and labels")
    labels = sorted(row[2] for row in rows)

    def quantile(probability):
        position = (len(labels) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        return labels[lower] + (labels[upper] - labels[lower]) * (position - lower)

    return {
        "count": len(rows),
        "source_segment_indices": [row[0] for row in rows],
        "window_center_s": [row[1] for row in rows],
        "range_km": [row[2] for row in rows],
        "identity_sha256": hashlib.sha256(json.dumps(rows, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
        "time_s": {"min": min(row[1] for row in rows), "max": max(row[1] for row in rows)},
        "range_km_summary": {"min": labels[0], "q25": quantile(0.25), "median": quantile(0.5), "q75": quantile(0.75), "max": labels[-1]},
    }


def range_coverage(training: dict, heldout: dict) -> dict:
    """Descriptive label-range coverage; never used for subset or model selection."""
    lower, upper = training["range_km_summary"]["min"], training["range_km_summary"]["max"]
    count = sum(lower <= label <= upper for label in heldout["range_km"])
    return {"within_train_range_count": count, "within_train_range_fraction": count / heldout["count"]}


def dataset_split_inventory(dataset, indices) -> dict:
    if not indices or len(set(indices)) != len(indices) or any(type(index) is not int or not 0 <= index < len(dataset) for index in indices):
        raise ValueError("Split inventory indices must be unique valid dataset positions")
    return split_inventory((dataset.source_segment_idx[index], dataset.window_center_s[index], dataset.labels[index]) for index in indices)


def training_inventory(bundle) -> dict:
    train = dataset_split_inventory(bundle.dataset, bundle.train_idx)
    validation = bundle.val_dataset
    if isinstance(validation, Subset):
        val = dataset_split_inventory(validation.dataset, list(validation.indices))
    else:
        val = dataset_split_inventory(validation, bundle.val_idx)
    if set(train["source_segment_indices"]) & set(val["source_segment_indices"]):
        raise ValueError("Training inventory contains overlapping train/validation source identities")
    return {"schema_version": 1, "train": train, "validation": val, "validation_coverage": range_coverage(train, val)}


def train(args: argparse.Namespace, bindings: MethodBindings) -> None:
    if args.no_resume and args.resume_checkpoint is not None:
        raise ValueError("--no-resume and --resume-checkpoint are mutually exclusive.")
    controlled = getattr(args, "controlled_batches", False)
    requested_updates = getattr(args, "optimizer_updates", None)
    validation_interval = getattr(args, "validation_interval_updates", None)
    if not controlled and (requested_updates is not None or validation_interval is not None):
        raise ValueError("Update budgets require --controlled-batches")
    if requested_updates is None:
        if type(args.epochs) is not int or args.epochs <= 0 or validation_interval is not None:
            raise ValueError("Epoch budgets require positive --epochs and no update validation interval")
    elif (args.epochs is not None or type(requested_updates) is not int or requested_updates <= 0
          or type(validation_interval) is not int or validation_interval <= 0 or requested_updates % validation_interval):
        raise ValueError("Update budgets require epochs=None and updates divisible by a positive validation interval")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers nonnegative.")
    if not math.isfinite(args.val_fraction) or not 0 < args.val_fraction < 1:
        raise ValueError("val_fraction must be finite and strictly between 0 and 1.")
    for key in ("lr", "weight_decay", "dropout", "huber_beta", "balance_loss_weight"):
        if not math.isfinite(getattr(args, key)):
            raise ValueError(f"{key} must be finite.")
    if args.lr <= 0 or args.weight_decay < 0 or args.huber_beta < 0 or not 0 <= args.dropout < 1 or args.base_channels <= 0:
        raise ValueError("Invalid optimizer, loss, or model settings.")
    if isinstance(args.balance_loss_weight, bool) or args.balance_loss_weight < 0:
        raise ValueError("balance_loss_weight must be finite and nonnegative")
    seed_everything(args.seed)
    resume_path = resolve_resume_checkpoint(args)
    if controlled and resume_path is not None:
        raise ValueError("Controlled training cannot resume: complete sampler/RNG state is not stored; use --no-resume with a new output directory")
    resume_checkpoint = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=args.device)
        if "budget_plan" in resume_checkpoint:
            raise ValueError("Controlled checkpoints cannot resume: complete sampler/RNG state is not stored")
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
    data_bundle, sampling = select_training_subset(data_bundle, args.train_fraction, args.seed)
    inventory = training_inventory(data_bundle)
    sources = source_identity(train_paths, val_paths)
    scaling_policy = {"reference_samples": args.scale_reference_samples,
                      "reference_experts": args.scale_reference_experts,
                      "exponent": args.scale_exponent, "max_experts": args.scale_max_experts}
    input_shape = data_bundle.input_shape
    if resume_checkpoint is not None:
        if resume_checkpoint["balance_loss_weight"] != args.balance_loss_weight:
            raise ValueError("Resume routing balance loss weight differs from checkpoint")
        for key, actual in (("source_identity", sources), ("sampling", sampling),
                            ("train_fraction", args.train_fraction), ("training_inventory", inventory)):
            if resume_checkpoint[key] != actual:
                raise ValueError(f"Resume training data differs from checkpoint: {key}.")
        require_matching_input_signature(resume_checkpoint["input_signature"],
                                         data_bundle.input_signature, "Resume checkpoint/data")
        for key, actual in (("train_indices", data_bundle.train_idx),
                            ("val_indices", data_bundle.val_idx),
                            ("split_mode", data_bundle.split_mode), ("seed", args.seed)):
            if resume_checkpoint[key] != actual:
                raise ValueError(f"Resume data split differs from checkpoint: {key}.")
    preprocessing_metadata = {}
    if bindings.method == "RBD_SCM":
        state = bindings.data.prepare_preprocessing(
            data_bundle, sampling,
            saved_state=None if resume_checkpoint is None else resume_checkpoint["input_preprocessing"],
        )
        preprocessing_metadata = {"input_preprocessing": state}
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

    budget_plan = None
    sampler_plan = None
    train_dataset = data_bundle.train_dataset
    epoch_sampler_plan = controlled_sampler_plan(args.seed)
    train_loader_options = {"generator": torch.Generator().manual_seed(epoch_sampler_plan["train_seed"])}
    val_loader_options = {"generator": torch.Generator().manual_seed(epoch_sampler_plan["validation_seed"])}
    if controlled:
        budget_plan = controlled_budget_plan(epochs=args.epochs, optimizer_updates=requested_updates,
                                             validation_interval_updates=validation_interval,
                                             train_count=train_count, batch_size=args.batch_size)
        sampler_plan = controlled_sampler_plan(args.seed)
        source_ids = [data_bundle.dataset.source_segment_idx[index] for index in data_bundle.train_idx]
        train_dataset = IndexedTrainingDataset(train_dataset, source_ids)
        train_loader_options = {"drop_last": True, "generator": torch.Generator().manual_seed(sampler_plan["train_seed"])}
        val_loader_options = {"generator": torch.Generator().manual_seed(sampler_plan["validation_seed"])}
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        **train_loader_options,
    )
    val_loader = DataLoader(
        data_bundle.val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        **val_loader_options,
    )

    _, feature_count, freq_bins = input_shape
    if resume_checkpoint is None:
        model_name = args.model
        config = bindings.model.build_model_config(
            model_name,
            input_shape,
            base_channels=args.base_channels,
            dropout=args.dropout,
            train_count=train_count,
            scaling_policy=scaling_policy,
        )
    else:
        model_name, config = bindings.model.model_config_from_checkpoint(resume_checkpoint)
        bindings.check_config_shape(config, input_shape)
        if config.train_count != train_count or config.scaling_policy != scaling_policy:
            raise ValueError("Resume training count or capacity policy differs from checkpoint")
        if config.base_channels != args.base_channels or config.dropout != args.dropout:
            raise ValueError("Resume model width/dropout differs from checkpoint")
    device = torch.device(args.device)
    model = bindings.model.build_model(model_name, config).to(device)
    initial_state_digest = model_state_digest(model) if controlled else None
    scaling = bindings.model.resolution_metadata(config)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != config.parameter_count:
        raise ValueError("Constructed model parameter count differs from resolved configuration")
    huber_beta = args.huber_beta if resume_checkpoint is None else resume_checkpoint["huber_beta"]
    criterion = nn.SmoothL1Loss(beta=huber_beta)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_epoch = 1
    end_epoch = budget_plan["validation_checks"] if controlled else args.epochs
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

    optimizer_steps = 0
    for row in history:
        steps = row.get("optimizer_steps_epoch")
        if type(steps) is not int or steps <= 0 or row.get("optimizer_steps") != optimizer_steps + steps:
            raise ValueError("Resume history requires consistent optimizer step evidence")
        elapsed = row.get("epoch_elapsed_s")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Resume history requires positive epoch_elapsed_s evidence")
        optimizer_steps += steps

    # Additional epochs form a fresh cosine stage from requested --lr to zero.
    # Restore optimizer momentum, but never continue the old periodic LR curve.
    for group in optimizer.param_groups:
        group["lr"] = args.lr
        group["initial_lr"] = args.lr
    if controlled:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget_plan["total_optimizer_updates"])
        scheduler_plan = {"policy": "cosine_per_update_v1", "step_unit": "optimizer_update",
                          "total_updates": budget_plan["total_optimizer_updates"], "initial_lr": float(args.lr), "eta_min": 0.0}
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scheduler_plan = {"policy": "new_cosine_phase", "start_epoch": start_epoch,
                          "phase_epochs": args.epochs, "initial_lr": float(args.lr), "eta_min": 0.0}
    controlled_metadata = ({"budget_plan": budget_plan, "sampler_plan": sampler_plan,
                            "scheduler_plan": scheduler_plan, "initial_model_state_sha256": initial_state_digest}
                           if controlled else {})
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
        for key, actual in (("source_identity", sources), ("sampling", sampling),
                            ("scaling", scaling), ("train_indices", data_bundle.train_idx),
                            ("val_indices", data_bundle.val_idx), ("seed", args.seed),
                            ("training_inventory", inventory),
                            ("balance_loss_weight", args.balance_loss_weight),
                            ("model_config", bindings.model.serialize_model_config(config))):
            if best_checkpoint.get(key) != actual:
                raise ValueError(f"Historical best checkpoint differs from current training: {key}")

        for key, value in preprocessing_metadata.items():
            if best_checkpoint.get(key) != value:
                raise ValueError(f"Historical best checkpoint differs from current preprocessing: {key}")

    architecture_path = with_time_suffix(output_dir / f"{dataset_name}_architecture.json", run_suffix)
    architecture_path.write_text(json.dumps({
        "pack": "MoE_data_scale", "method": bindings.method, "model": model_name,
        "scaling": scaling, "sampling": sampling,
        "balance_loss_weight": args.balance_loss_weight,
        "training_inventory": inventory,
        "model_config": bindings.model.serialize_model_config(config),
        "input_signature": data_bundle.input_signature,
        **preprocessing_metadata,
        **controlled_metadata,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(f"Split mode: {split_mode}")
    print(f"Source files: {len(source_paths)}")
    print(f"Dataset: {args.data}")
    print(f"Model: {model_name}")
    print(f"Train/val samples: {train_count}/{val_count}")
    for split, values in (("Train", inventory["train"]), ("Validation", inventory["validation"])):
        print(f"{split} membership: sha256={values['identity_sha256']} | time_s={values['time_s']} | range_km={values['range_km_summary']}")
    print(f"Validation range coverage: {inventory['validation_coverage']}")
    print(f"Capacity: {scaling['selection_mode']} E={config.num_experts}, top-k={config.top_k} | "
          f"parameters {config.parameter_count} | policy {config.scaling_policy_version} (uncalibrated)")
    print(f"Architecture: {architecture_path}")
    print(f"Input: 2 x {feature_count} x {freq_bins} ({bindings.method})")
    print(f"Target normalization: mean={y_mean:.4f} km, std={y_std:.4f} km")
    print(f"Loss space: {loss_space} (SmoothL1 beta={huber_beta:g})")
    print(f"Device: {device}")
    print(f"Run suffix: {run_suffix}")
    if controlled:
        print(f"Controlled budget: {budget_plan}")
        print(f"Independent samplers: {sampler_plan}")
        print(f"Initial model state: {initial_state_digest}")
    if resume_checkpoint is not None:
        print(f"Resume checkpoint: {resume_path}")
        print(f"Resume epochs: {start_epoch}-{end_epoch}")
        print(f"Resume scheduler: new cosine phase from --lr={args.lr:g} to 0; optimizer momentum/history restored.")

    batch_stream = ControlledBatchStream(train_loader) if controlled else None
    lr_digest = hashlib.sha256()
    last_update_lr = None

    def after_controlled_update():
        nonlocal last_update_lr
        last_update_lr = float(optimizer.param_groups[0]["lr"])
        lr_digest.update((json.dumps(last_update_lr, allow_nan=False) + "\n").encode())
        scheduler.step()

    for epoch in range(start_epoch, end_epoch + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        record_first_lr = float(optimizer.param_groups[0]["lr"])
        record_loader = batch_stream.take_updates(budget_plan["validation_interval_updates"]) if controlled else train_loader
        train_metrics = run_epoch(
            model,
            record_loader,
            criterion,
            device,
            y_mean,
            y_std,
            loss_space,
            optimizer,
            on_optimizer_step=after_controlled_update if controlled else None,
            balance_loss_weight=args.balance_loss_weight,
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, y_mean, y_std, loss_space,
            balance_loss_weight=args.balance_loss_weight,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_elapsed_s = time.perf_counter() - epoch_started
        optimizer_steps_epoch = budget_plan["validation_interval_updates"] if controlled else len(train_loader)
        optimizer_steps += optimizer_steps_epoch
        epoch_lr = record_first_lr if controlled else float(optimizer.param_groups[0]["lr"])
        if not controlled:
            scheduler.step()
        training_progress = None
        controlled_record = {}
        if controlled:
            training_progress = {**batch_stream.progress(), "lr_sequence_sha256": lr_digest.hexdigest()}
            if training_progress["optimizer_steps"] != optimizer_steps:
                raise ValueError("Controlled stream and optimizer update counts disagree")
            controlled_record = {key: value for key, value in training_progress.items()
                                 if key not in {"seen_source_segment_indices", "data_pass_batch_sequence_sha256"}}
            controlled_record.update(record_unit="validation_checkpoint", validation_index=epoch,
                                     sample_presentations_record=optimizer_steps_epoch * args.batch_size,
                                     last_update_lr=last_update_lr)

        row = {
            "epoch": epoch,
            "lr": epoch_lr,
            "next_lr": float(optimizer.param_groups[0]["lr"]),
            "loss_space": loss_space,
            "epoch_elapsed_s": epoch_elapsed_s,
            "optimizer_steps_epoch": optimizer_steps_epoch,
            "optimizer_steps": optimizer_steps,
            **controlled_record,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"{'validation' if controlled else 'epoch'} {epoch:03d} | "
            f"train loss {row['train_loss']:.4f} mae {row['train_mae_km']:.3f} km | "
            f"val loss {row['val_loss']:.4f} mae {row['val_mae_km']:.3f} km "
            f"rmse {row['val_rmse_km']:.3f} km | {epoch_elapsed_s:.3f} s | optimizer steps {optimizer_steps}"
        )

        improved = val_metrics["rmse_km"] < best_rmse
        if improved:
            actual_best_path = best_path
        checkpoint = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "pack": "MoE_data_scale",
            "scaling": scaling,
            "sampling": sampling,
            "training_inventory": inventory,
            "source_identity": sources,
            "train_fraction": args.train_fraction,
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
            "balance_loss_weight": args.balance_loss_weight,
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
            **preprocessing_metadata,
            **controlled_metadata,
        }
        if controlled:
            checkpoint["training_progress"] = training_progress
        if improved:
            best_rmse = val_metrics["rmse_km"]
            checkpoint["best_rmse_km"] = best_rmse
            torch.save(checkpoint, best_path)
        torch.save(checkpoint, last_path)

        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Best checkpoint: {actual_best_path}")
    print(f"Last checkpoint: {last_path}")
