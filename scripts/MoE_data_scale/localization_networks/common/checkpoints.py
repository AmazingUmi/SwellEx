"""Strict contract for neural artifacts written by the current runtime."""
from __future__ import annotations

import math
import hashlib
import json
import re
from typing import Any


CHECKPOINT_VERSION = 1


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def validate_controlled_evidence(checkpoint: dict[str, Any], train_count: int) -> None:
    """New opt-in evidence is strict; historical epoch artifacts remain valid."""
    from .training_utils import controlled_budget_plan, controlled_sampler_plan

    required = {"budget_plan", "sampler_plan", "scheduler_plan", "initial_model_state_sha256",
                "training_progress", "history", "metrics", "epoch", "seed", "training_inventory",
                "scheduler_state", "optimizer_state"}
    if not required.issubset(checkpoint):
        raise ValueError("Controlled checkpoint is missing training-budget evidence")
    budget = checkpoint["budget_plan"]
    if not isinstance(budget, dict):
        raise ValueError("Controlled checkpoint budget_plan must be an object")
    expected = controlled_budget_plan(
        epochs=budget.get("requested_epochs"), optimizer_updates=budget.get("requested_optimizer_updates"),
        validation_interval_updates=budget.get("validation_interval_updates") if budget.get("mode") == "updates" else None,
        train_count=train_count, batch_size=budget.get("batch_size"),
    )
    if budget != expected or checkpoint["sampler_plan"] != controlled_sampler_plan(checkpoint["seed"]):
        raise ValueError("Controlled checkpoint budget or sampler recipe is inconsistent")
    scheduler = checkpoint["scheduler_plan"]
    if not isinstance(scheduler, dict):
        raise ValueError("Controlled checkpoint scheduler_plan must be an object")
    initial_lr = scheduler.get("initial_lr")
    if isinstance(initial_lr, bool) or not isinstance(initial_lr, (int, float)) or not math.isfinite(initial_lr) or initial_lr <= 0:
        raise ValueError("Controlled checkpoint initial learning rate must be positive")
    if scheduler != {"policy": "cosine_per_update_v1", "step_unit": "optimizer_update",
                     "total_updates": budget["total_optimizer_updates"], "initial_lr": initial_lr, "eta_min": 0.0}:
        raise ValueError("Controlled checkpoint scheduler differs from its update budget")
    if not _sha256(checkpoint["initial_model_state_sha256"]):
        raise ValueError("Controlled checkpoint lacks a valid initial model-state identity")
    history = checkpoint["history"]
    if not isinstance(history, list) or not history or len(history) > budget["validation_checks"] or checkpoint["epoch"] != len(history):
        raise ValueError("Controlled checkpoint validation history is inconsistent")
    unique_seen = 0
    next_lr = float(initial_lr)
    lr_digest = hashlib.sha256()
    total_updates = budget["total_optimizer_updates"]
    for index, row in enumerate(history, 1):
        updates = index * budget["validation_interval_updates"]
        expected_row = {
            "epoch": index, "validation_index": index, "record_unit": "validation_checkpoint",
            "optimizer_steps_epoch": budget["validation_interval_updates"], "optimizer_steps": updates,
            "sample_presentations_record": budget["validation_interval_updates"] * budget["batch_size"],
            "sample_presentations": updates * budget["batch_size"],
            "data_passes_completed": updates // budget["steps_per_data_pass"],
            "updates_in_current_pass": updates % budget["steps_per_data_pass"],
        }
        if not isinstance(row, dict) or any(type(row.get(key)) is not type(value) or row[key] != value
                                            for key, value in expected_row.items()):
            raise ValueError("Controlled validation records disagree with update/sample/data-pass counters")
        seen = row.get("unique_train_sources_seen")
        if type(seen) is not int or not max(unique_seen, budget["batch_size"]) <= seen <= min(train_count, updates * budget["batch_size"]):
            raise ValueError("Controlled unique-source counter is invalid")
        unique_seen = seen
        first_lr = next_lr
        for update in range(updates - budget["validation_interval_updates"] + 1, updates + 1):
            last_lr = next_lr
            lr_digest.update((json.dumps(last_lr, allow_nan=False) + "\n").encode())
            # Replay the frozen CosineAnnealingLR recurrence, including its
            # floating-point order, to verify the actual per-update LR digest.
            next_lr = ((1 + math.cos(math.pi * update / total_updates))
                       / (1 + math.cos(math.pi * (update - 1) / total_updates))) * last_lr
        for key, expected_lr in (("lr", first_lr), ("last_update_lr", last_lr), ("next_lr", next_lr)):
            value = row.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not math.isclose(value, expected_lr, rel_tol=1e-12, abs_tol=1e-15)):
                raise ValueError("Controlled history learning rates differ from the per-update cosine schedule")
        if row.get("lr_sequence_sha256") != lr_digest.hexdigest():
            raise ValueError("Controlled history learning-rate trajectory digest differs from the schedule")
        for key in ("batch_sequence_sha256", "record_batch_sequence_sha256", "lr_sequence_sha256", "seen_source_identity_sha256"):
            if not _sha256(row.get(key)):
                raise ValueError(f"Controlled record lacks {key}")
    scheduler_state = checkpoint["scheduler_state"]
    expected_scheduler_state = {"T_max": total_updates, "eta_min": 0.0, "base_lrs": [initial_lr],
                                "last_epoch": updates, "_step_count": updates + 1, "_last_lr": [next_lr]}
    if not isinstance(scheduler_state, dict) or any(scheduler_state.get(key) != value for key, value in expected_scheduler_state.items()):
        raise ValueError("Controlled scheduler state differs from the completed optimizer updates")
    optimizer_state = checkpoint["optimizer_state"]
    groups = optimizer_state.get("param_groups") if isinstance(optimizer_state, dict) else None
    if not isinstance(groups, list) or len(groups) != 1 or groups[0].get("lr") != next_lr or groups[0].get("initial_lr") != initial_lr:
        raise ValueError("Controlled optimizer learning rate differs from the scheduler")
    progress = checkpoint["training_progress"]
    progress_keys = {"optimizer_steps", "sample_presentations", "data_passes_completed", "updates_in_current_pass",
                     "unique_train_sources_seen", "seen_source_segment_indices", "seen_source_identity_sha256",
                     "batch_sequence_sha256", "record_batch_sequence_sha256", "lr_sequence_sha256",
                     "data_pass_batch_sequence_sha256"}
    if not isinstance(progress, dict) or set(progress) != progress_keys:
        raise ValueError("Controlled training progress schema is incomplete")
    if any(progress[key] != history[-1][key] for key in progress_keys - {"seen_source_segment_indices", "data_pass_batch_sequence_sha256"}) or checkpoint["metrics"] != history[-1]:
        raise ValueError("Controlled progress differs from the latest validation record")
    pass_digests = progress["data_pass_batch_sequence_sha256"]
    if (not isinstance(pass_digests, list) or len(pass_digests) != progress["data_passes_completed"]
            or any(not _sha256(value) for value in pass_digests)):
        raise ValueError("Controlled data-pass batch-prefix evidence is invalid")
    for row in history:
        if "data_pass_batch_sequence_sha256" in row:
            raise ValueError("Data-pass batch-prefix lists belong only in training_progress")
        if (row["updates_in_current_pass"] == 0
                and pass_digests[row["data_passes_completed"] - 1] != row["batch_sequence_sha256"]):
            raise ValueError("Controlled data-pass batch prefix differs from validation history")
    seen = progress["seen_source_segment_indices"]
    if (not isinstance(seen, list) or not seen or any(type(value) is not int or value <= 0 for value in seen)
            or seen != sorted(set(seen)) or len(seen) != unique_seen):
        raise ValueError("Controlled seen-source inventory is invalid")
    inventory = checkpoint["training_inventory"]
    if not isinstance(inventory, dict) or not isinstance(inventory.get("train"), dict):
        raise ValueError("Controlled checkpoint is missing its eligible training inventory")
    eligible = inventory["train"].get("source_segment_indices", [])
    if not set(seen).issubset(eligible):
        raise ValueError("Controlled batch stream includes ineligible training sources")
    expected_hash = hashlib.sha256(json.dumps(seen, separators=(",", ":")).encode()).hexdigest()
    if progress["seen_source_identity_sha256"] != expected_hash:
        raise ValueError("Controlled seen-source identity hash differs from its members")


def validate_checkpoint(checkpoint: dict[str, Any], *, method: str, model_name: str, dataset_name: str) -> None:
    required = {"checkpoint_version", "pack", "scaling", "sampling", "source_identity", "train_fraction", "method", "model_name", "dataset_name", "model_config",
                "model_state", "input_signature", "input_normalized_per_sample",
                "target_mean_km", "target_std_km", "loss_space", "huber_beta", "balance_loss_weight"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        missing = sorted(required - checkpoint.keys()) if isinstance(checkpoint, dict) else sorted(required)
        raise ValueError(f"Unsupported checkpoint: missing contract fields {missing}; regenerate with current scripts.")
    if checkpoint["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version; regenerate with current scripts.")
    if checkpoint["pack"] != "MoE_data_scale":
        raise ValueError("Checkpoint pack must be MoE_data_scale")
    from importlib import import_module
    if method not in {"RBD", "ELM", "SCM", "RBD_SCM"}:
        raise ValueError("Invalid checkpoint method")
    registry = import_module(f"scripts.MoE_data_scale.localization_networks.{method.lower()}.network.model")
    _, config = registry.model_config_from_checkpoint(checkpoint)
    from .input_contract import validate_saved_input_signature
    shape_field = "input_features" if method == "RBD_SCM" else ("input_elements" if method == "RBD" else "input_pairs")
    validate_saved_input_signature(checkpoint["input_signature"],
                                   (2, getattr(config, shape_field), config.input_freq_bins), method)
    if checkpoint["scaling"] != registry.resolution_metadata(config):
        raise ValueError("Checkpoint scaling metadata differs from resolved model configuration")
    sampling = checkpoint["sampling"]
    if not isinstance(sampling, dict) or sampling.get("train_count") != config.train_count or sampling.get("train_fraction") != checkpoint["train_fraction"]:
        raise ValueError("Checkpoint training sample selection is inconsistent")
    if method == "RBD_SCM":
        from scripts.MoE_data_scale.localization_networks.rbd_scm.contract import validate_preprocessing
        state = checkpoint.get("input_preprocessing")
        validate_preprocessing(state, checkpoint["input_signature"])
        if (state["fit_sample_count"] != config.train_count
                or state["fit_source_identity_sha256"] != sampling.get("source_identity_sha256")):
            raise ValueError("Checkpoint block_rms fitted membership differs from training selection")
    fraction = checkpoint["train_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Checkpoint train_fraction must be in (0, 1]")
    for key, expected in (("method", method), ("model_name", model_name), ("dataset_name", dataset_name)):
        if checkpoint[key] != expected:
            raise ValueError(f"Checkpoint {key} '{checkpoint[key]}' does not match requested '{expected}'.")
    if not isinstance(checkpoint["input_signature"], dict) or checkpoint["input_signature"].get("method") != method:
        raise ValueError("Checkpoint input_signature is missing or inconsistent with its method.")
    if not isinstance(checkpoint["input_normalized_per_sample"], bool):
        raise ValueError("Checkpoint input normalization policy must be boolean.")
    for key in ("target_mean_km", "target_std_km", "huber_beta"):
        if not math.isfinite(float(checkpoint[key])):
            raise ValueError(f"Checkpoint {key} must be finite.")
    if float(checkpoint["target_std_km"]) <= 0 or float(checkpoint["huber_beta"]) < 0:
        raise ValueError("Checkpoint target std must be positive and Huber beta nonnegative.")
    if checkpoint["loss_space"] not in {"km", "normalized"}:
        raise ValueError("Checkpoint loss_space is invalid.")
    weight = checkpoint["balance_loss_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("Checkpoint balance_loss_weight must be finite and nonnegative")
    if any(key in checkpoint for key in ("budget_plan", "sampler_plan", "initial_model_state_sha256", "training_progress")):
        validate_controlled_evidence(checkpoint, config.train_count)
