"""Shared training utilities for range regression."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .paths import latest_time_suffixed_path, safe_name, train_output_dir


def seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 4294967295.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def target_stats(labels: torch.Tensor) -> tuple[float, float]:
    if labels.numel() == 0 or not bool(torch.isfinite(labels).all()):
        raise ValueError("Target statistics require nonempty finite labels.")
    mean = float(labels.mean())
    std = float(labels.std(unbiased=False))
    if not math.isfinite(std) or std < 1.0e-6:
        std = 1.0
    return mean, std


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.no_resume and args.resume_checkpoint is not None:
        raise ValueError("--no-resume and --resume-checkpoint are mutually exclusive.")
    if args.no_resume:
        return None
    if args.resume_checkpoint is not None:
        return args.resume_checkpoint

    output_dir = train_output_dir(args.output_dir, args.model)
    dataset_name = safe_name(args.data)
    fixed_last_path = output_dir / f"{dataset_name}_last.pt"
    if fixed_last_path.exists():
        return fixed_last_path
    latest_path = latest_time_suffixed_path(fixed_last_path)
    if latest_path is not None:
        return latest_path
    return None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    y_mean: float,
    y_std: float,
    loss_space: str = "normalized",
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    if not math.isfinite(y_mean) or not math.isfinite(y_std) or y_std <= 0:
        raise ValueError("Target normalization must be finite with positive std.")
    if loss_space not in {"normalized", "km"}:
        raise ValueError(f"Unsupported loss_space: {loss_space}")
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_count = 0

    for x, y_km in loader:
        x = x.to(device, non_blocking=True)
        y_km = y_km.to(device, non_blocking=True)
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y_km).all()):
            raise ValueError("Epoch input features/labels contain nonfinite values.")
        y_norm = (y_km - y_mean) / y_std

        with torch.set_grad_enabled(training):
            pred_norm = model(x)
            if pred_norm.shape != y_km.shape or not bool(torch.isfinite(pred_norm).all()):
                raise ValueError("Model must return finite normalized predictions shaped [batch,1].")
            pred_km = pred_norm * y_std + y_mean
            if loss_space == "normalized":
                loss = criterion(pred_norm, y_norm)
            elif loss_space == "km":
                loss = criterion(pred_km, y_km)
            else:
                raise ValueError(f"Unsupported loss_space: {loss_space}")

            if not bool(torch.isfinite(loss)):
                raise ValueError("Epoch loss is nonfinite.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = x.size(0)
        pred_km = pred_km.detach()
        err = pred_km - y_km
        total_loss += float(loss.detach()) * batch_size
        total_abs_error += float(err.abs().sum())
        total_sq_error += float(err.square().sum())
        total_count += batch_size

    if total_count == 0:
        raise ValueError("Cannot compute metrics for an empty epoch.")
    rmse = math.sqrt(total_sq_error / total_count)
    return {
        "loss": total_loss / total_count,
        "mae_km": total_abs_error / total_count,
        "rmse_km": rmse,
    }
