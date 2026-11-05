"""Shared training utilities for range regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
    *,
    on_optimizer_step=None,
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
                if on_optimizer_step is not None:
                    on_optimizer_step()

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


CONTROLLED_BUDGET_VERSION = "controlled-budget-v1"
CONTROLLED_SAMPLER_VERSION = "independent-torch-generators-v1"


def controlled_budget_plan(*, epochs, optimizer_updates, validation_interval_updates,
                           train_count: int, batch_size: int) -> dict:
    for key, value in (("train_count", train_count), ("batch_size", batch_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    steps_per_pass = train_count // batch_size
    if not steps_per_pass:
        raise ValueError("Controlled batches require at least one complete training batch")
    if optimizer_updates is None:
        if type(epochs) is not int or epochs <= 0 or validation_interval_updates is not None:
            raise ValueError("Controlled epochs require positive epochs and no update validation interval")
        mode, total, interval = "epochs", epochs * steps_per_pass, steps_per_pass
    else:
        if epochs is not None or type(optimizer_updates) is not int or optimizer_updates <= 0:
            raise ValueError("Optimizer-update budgets require positive updates and epochs=None")
        if type(validation_interval_updates) is not int or validation_interval_updates <= 0 or optimizer_updates % validation_interval_updates:
            raise ValueError("Optimizer updates must be divisible by a positive validation interval")
        mode, total, interval = "updates", optimizer_updates, validation_interval_updates
    if total < math.ceil(train_count / batch_size):
        raise ValueError("Controlled update budget cannot present every eligible training member even once")
    return {"version": CONTROLLED_BUDGET_VERSION, "mode": mode,
            "requested_epochs": epochs, "requested_optimizer_updates": optimizer_updates,
            "total_optimizer_updates": total, "validation_interval_updates": interval,
            "validation_checks": total // interval, "batch_size": batch_size, "drop_last": True,
            "steps_per_data_pass": steps_per_pass, "controlled_batches": True}


def controlled_sampler_plan(seed: int) -> dict:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Controlled sampler requires a uint32 seed")
    return {"version": CONTROLLED_SAMPLER_VERSION, "train_seed": seed + 1000003,
            "validation_seed": seed + 2000003, "batch_digest": "sha256-json-source-id-batches-v1"}


class IndexedTrainingDataset(Dataset):
    """Expose physical identities only to the controlled batch evidence stream."""

    def __init__(self, dataset, source_ids):
        if len(dataset) != len(source_ids) or len(set(source_ids)) != len(source_ids):
            raise ValueError("Controlled dataset requires one unique source ID per row")
        self.dataset = dataset
        self.source_ids = list(source_ids)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, self.source_ids[index]


class ControlledBatchStream:
    """One independently shuffled, drop-last stream spanning validation records."""

    def __init__(self, loader: DataLoader):
        if not loader.drop_last or loader.generator is None or len(loader) < 1:
            raise ValueError("Controlled stream requires a nonempty independent-generator drop-last loader")
        self.loader = loader
        self.iterator = None
        self.steps = 0
        self.samples = 0
        self.completed_passes = 0
        self.updates_in_pass = 0
        self.seen_sources = set()
        self.digest = hashlib.sha256()
        self.record_digest = hashlib.sha256()
        self.data_pass_digests = []

    def take_updates(self, count: int):
        if type(count) is not int or count <= 0:
            raise ValueError("Controlled record must contain a positive number of updates")
        self.record_digest = hashlib.sha256()
        for _ in range(count):
            if self.iterator is None:
                self.iterator = iter(self.loader)
            x, y, identities = next(self.iterator)
            identities = [int(value) for value in identities.tolist()]
            if len(identities) != self.loader.batch_size:
                raise ValueError("Controlled stream yielded an incomplete batch")
            encoded = (json.dumps(identities, separators=(",", ":")) + "\n").encode("utf-8")
            self.digest.update(encoded)
            self.record_digest.update(encoded)
            self.seen_sources.update(identities)
            self.steps += 1
            self.samples += len(identities)
            self.updates_in_pass += 1
            if self.updates_in_pass == len(self.loader):
                # Exhaust the drop-last iterator, including its discarded tail,
                # so a data pass has exactly the normal DataLoader RNG behavior.
                try:
                    next(self.iterator)
                except StopIteration:
                    pass
                else:
                    raise ValueError("Controlled loader yielded more batches than declared")
                self.iterator = None
                self.completed_passes += 1
                self.updates_in_pass = 0
                self.data_pass_digests.append(self.digest.hexdigest())
            yield x, y

    def progress(self) -> dict:
        seen = sorted(self.seen_sources)
        return {"optimizer_steps": self.steps, "sample_presentations": self.samples,
                "data_passes_completed": self.completed_passes, "updates_in_current_pass": self.updates_in_pass,
                "unique_train_sources_seen": len(seen), "seen_source_segment_indices": seen,
                "seen_source_identity_sha256": hashlib.sha256(json.dumps(seen, separators=(",", ":")).encode()).hexdigest(),
                "batch_sequence_sha256": self.digest.hexdigest(),
                "record_batch_sequence_sha256": self.record_digest.hexdigest(),
                "data_pass_batch_sequence_sha256": list(self.data_pass_digests)}


def model_state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update((json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")) + "\n").encode())
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()
