"""Small staged experiment, with isolated candidates and fixed stage-one scales."""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import math
import os
import time

import torch
from torch import nn

from .checkpoints import save_checkpoint, state_digest, validate_metadata
from .controller import ControllerPolicy, assess_candidate, propose_add, select_candidate
from .model import ContinualMoE, ModelConfig


def require_cuda(device):
    # Replay/test-isolation checks require repeatable CUDA kernels, not only seeds.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    chosen = torch.device(device)
    if chosen.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MoE_learn requires the explicit CUDA device; CPU fallback is forbidden")
    torch.ones(1, device=chosen)
    return chosen


def tensors(x, y):
    features = torch.as_tensor(x, dtype=torch.float32).detach().cpu().contiguous()
    labels = torch.as_tensor(y, dtype=torch.float32).detach().cpu().contiguous()
    if features.ndim != 4 or not len(features) or labels.shape != (len(features), 1):
        raise ValueError("Training data must be nonempty [N,2,H,F] / [N,1]")
    if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(labels).all()):
        raise ValueError("Nonfinite features or range labels")
    return features, labels


@torch.no_grad()
def evaluate(model, dataset, target, device, batch_size, *, prediction_path=None, source_ids=None, window_centers=None):
    model.eval()
    x, y = dataset
    errors = []
    predictions = []
    load = torch.zeros(model.num_experts, dtype=torch.long)
    entropy = 0.0
    for start in range(0, len(x), batch_size):
        inputs = x[start:start + batch_size].to(device)
        prediction, aux = model.forward_with_aux(inputs)
        prediction = prediction * target["std_km"] + target["mean_km"]
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError("Nonfinite candidate predictions")
        errors.append(prediction.cpu() - y[start:start + batch_size])
        if prediction_path is not None:
            predictions.extend(prediction.cpu().flatten().tolist())
        load += aux["expert_load"].cpu()
        entropy += float(aux["router_entropy"]) * len(inputs)
    error = torch.cat(errors)
    if prediction_path is not None:
        if source_ids is None or len(source_ids) != len(x):
            raise ValueError("Prediction export requires fixed test source identities")
        with prediction_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_segment_idx", "window_center_s", "true_range_km", "pred_range_km", "error_m"])
            for index, prediction in enumerate(predictions):
                truth = float(y[index, 0])
                writer.writerow([source_ids[index], "" if window_centers is None else window_centers[index],
                                 truth, prediction, 1000 * (prediction - truth)])
    return {"count": len(x), "mae_km": float(error.abs().mean()), "rmse_km": float(error.square().mean().sqrt()),
            "expert_load": load.tolist(), "router_entropy": entropy / len(x)}


def merge_datasets(datasets):
    return tuple(torch.cat([item[index] for item in datasets], dim=0) for index in (0, 1))


def optimizer_groups(model, settings, *, phase, new_expert_id=None):
    """Build a disjoint, exhaustive parameter partition with explicit learning rates."""
    if phase not in {"initial", "adapt", "warmup"}:
        raise ValueError("Unknown training phase")
    if new_expert_id is not None and (type(new_expert_id) is not int or not 0 <= new_expert_id < model.num_experts):
        raise ValueError("Invalid new expert identity")
    if phase == "warmup" and new_expert_id is None:
        raise ValueError("Warmup requires a new expert")
    if phase == "initial" and new_expert_id is not None:
        raise ValueError("Initial training cannot designate a new expert")
    groups, metadata = [], []

    def add(role, parameters, learning_rate, expert_ids):
        parameters = list(parameters)
        if not parameters:
            return
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Optimizer learning rates must be finite and positive")
        groups.append({"params": parameters, "lr": learning_rate, "name": role})
        metadata.append({"role": role, "learning_rate": learning_rate,
                         "parameter_count": sum(p.numel() for p in parameters), "expert_ids": expert_ids})

    if phase == "initial":
        if model.stem_frozen or model.frozen_experts or any(not p.requires_grad for p in model.parameters()):
            raise ValueError("Initial training requires a fresh fully trainable model")
        add("initial", model.parameters(), settings["learning_rate"], list(range(model.num_experts)))
    elif phase == "warmup":
        model.configure_training((new_expert_id,), train_router=False)
        add("new_expert", model.experts[new_expert_id].parameters(), settings["new_expert_learning_rate"], [new_expert_id])
    else:
        model.configure_training(range(model.num_experts))
        old_ids = [index for index in range(model.num_experts) if index != new_expert_id]
        add("old_experts", (p for index in old_ids for p in model.experts[index].parameters()),
            settings["old_expert_learning_rate"], old_ids)
        if new_expert_id is not None:
            add("new_expert", model.experts[new_expert_id].parameters(), settings["new_expert_learning_rate"], [new_expert_id])
        add("router", model.router.parameters(), settings["router_learning_rate"], [])
    actual = [id(p) for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Optimizer groups must contain every trainable parameter exactly once")
    return groups, metadata


def train_epochs(model, dataset, validation, target, device, settings, balance_weight, epochs, seed, *, phase="initial", new_expert_id=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sampler = torch.Generator().manual_seed(seed)
    groups, group_metadata = optimizer_groups(model, settings, phase=phase, new_expert_id=new_expert_id)
    parameters = [p for group in groups for p in group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])
    criterion = nn.SmoothL1Loss(beta=1.0)
    batch_size = settings["batch_size"]
    x, y = dataset
    history, best_state, best_rmse = [], None, float("inf")
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_balance = total_entropy = 0.0
        loads = torch.zeros(model.num_experts, dtype=torch.long)
        steps = 0
        for indices in torch.randperm(len(x), generator=sampler).split(batch_size):
            inputs, labels = x[indices].to(device), y[indices].to(device)
            normalized = (labels - target["mean_km"]) / target["std_km"]
            if phase != "warmup":
                predictions, aux = model.forward_with_aux(inputs)
                balance = aux["balance_loss"]
                loads += aux["expert_load"].detach().cpu()
                total_entropy += float(aux["router_entropy"].detach()) * len(indices)
            else:
                predictions = model.expert_prediction(inputs, new_expert_id)
                balance = predictions.sum() * 0.0
                loads[new_expert_id] += len(indices)
            regression = criterion(predictions, normalized)
            objective = regression + balance_weight * balance
            if not bool(torch.isfinite(objective)):
                raise ValueError("Nonfinite training objective")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total_loss += float(regression.detach()) * len(indices)
            total_balance += float(balance.detach()) * len(indices)
            steps += 1
        metrics = evaluate(model, validation, target, device, batch_size) if phase != "warmup" else None
        history.append({"epoch": epoch, "regression_loss": total_loss / len(x), "balance_loss": total_balance / len(x),
                        "router_entropy": total_entropy / len(x), "expert_load": loads.tolist(), "optimizer_steps": steps,
                        "validation": metrics})
        if metrics is not None and metrics["rmse_km"] < best_rmse:
            best_rmse = metrics["rmse_km"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(f"  {phase} epoch {epoch}/{epochs}: loss={total_loss / len(x):.6f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"phase": phase, "optimizer_groups": group_metadata,
            "epochs": epochs, "optimizer_steps": sum(row["optimizer_steps"] for row in history),
            "sample_presentations": epochs * len(x), "training_rows": len(x),
            "wall_seconds": time.perf_counter() - start_time, "history": history}


@torch.no_grad()
def expert_updates(before, after):
    """Observed drift of existing experts; learning-rate ratios do not bound this drift."""
    updates = []
    for index, old in enumerate(before.experts):
        squared_delta = squared_reference = maximum = 0.0
        for first, second in zip(old.parameters(), after.experts[index].parameters()):
            reference = first.detach().double()
            difference = second.detach().double() - reference
            squared_delta += float(difference.square().sum())
            squared_reference += float(reference.square().sum())
            maximum = max(maximum, float(difference.abs().max()))
        updates.append({"expert_id": index, "l2_delta": math.sqrt(squared_delta),
                        "relative_l2_delta": math.sqrt(squared_delta) / max(math.sqrt(squared_reference), 1e-12),
                        "max_abs_delta": maximum, "changed": squared_delta > 0})
    return updates


@torch.no_grad()
def descriptors(model, features, device, batch_size):
    model.eval()
    return torch.cat([model.descriptor(batch.to(device)).cpu() for batch in features.split(batch_size)])


def fit_descriptor(model, features, device, batch_size):
    values = descriptors(model, features, device, batch_size)
    return {"mean": values.mean(dim=0).tolist(), "variance": values.var(dim=0, correction=0).tolist(),
            "count": len(values), "fit_stage": 1, "version": "frozen-stem-magnitude-mean-std-v1"}


def novelty_score(model, features, reference, device, batch_size):
    values = descriptors(model, features, device, batch_size)
    mean = torch.tensor(reference["mean"])
    variance = torch.tensor(reference["variance"])
    # Reference-relative mean displacement, protected against near-constant dimensions.
    floor = max(float(variance.mean()) * 0.01, 1e-6)
    score = ((values.mean(dim=0) - mean).square() / variance.clamp_min(floor)).mean().sqrt()
    return float(score)


class ReplayBuffer:
    """Reservoir over optimizer-training rows only; never validation/test rows."""
    def __init__(self, capacity, seed):
        self.capacity, self.seen = capacity, 0
        self.generator = torch.Generator().manual_seed(seed)
        self.rows, self.identities = [], []

    def add(self, dataset, stage, source_ids):
        if len(source_ids) != len(dataset[0]):
            raise ValueError("Replay source identities do not match optimizer rows")
        if not self.capacity:
            return
        for x, y, source_id in zip(*dataset, source_ids):
            self.seen += 1
            index = len(self.rows) if len(self.rows) < self.capacity else int(torch.randint(self.seen, (), generator=self.generator))
            if index < self.capacity:
                row, identity = (x.clone(), y.clone()), {"stage": stage, "source_id": source_id}
                if index == len(self.rows):
                    self.rows.append(row)
                    self.identities.append(identity)
                else:
                    self.rows[index], self.identities[index] = row, identity

    def dataset(self):
        return tuple(torch.stack([row[index] for row in self.rows]) for index in (0, 1)) if self.rows else None


def run_stages(config, prepared, output_dir, *, device):
    settings, continual = config["training"], config["continual"]
    policy = ControllerPolicy(**config["controller"])
    metadata = prepared["metadata"]
    raw_stages = prepared["stages"]
    if len(raw_stages) != continual["stages"]:
        raise ValueError("Configured stage count differs from prepared data")
    stages = [{"train": tensors(stage["train_x"], stage["train_y"]), "val": tensors(stage["val_x"], stage["val_y"])} for stage in raw_stages]
    test = tensors(prepared["test_x"], prepared["test_y"])
    shape = stages[0]["train"][0].shape
    model_config = ModelConfig(shape[2], shape[3], settings["base_channels"], continual["initial_experts"])
    validate_metadata(metadata, model_config)
    for number, stage in enumerate(stages):
        for name, field in (("train", "train_source_ids"), ("val", "val_source_ids")):
            if len(stage[name][0]) != len(metadata["stages"][number][field]) or tuple(stage[name][0].shape[1:]) != tuple(shape[1:]):
                raise ValueError("Prepared tensors disagree with stage membership/input shape")
    if len(test[0]) != len(metadata["test_source_ids"]) or tuple(test[0].shape[1:]) != tuple(shape[1:]):
        raise ValueError("Test tensors disagree with metadata")
    seed, batch_size = settings["seed"], settings["batch_size"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    deployed = ContinualMoE(model_config).to(device)
    first_labels = stages[0]["train"][1]
    target = {"mean_km": float(first_labels.mean()), "std_km": max(float(first_labels.std(correction=0)), 1e-6),
              "fit_stage": 1, "fit_sample_count": len(first_labels),
              "fit_source_identity_sha256": metadata["preprocessing"]["fit_source_identity_sha256"]}
    print(f"Stage 1: train {len(first_labels)} rows with {deployed.num_experts} experts", flush=True)
    initial = train_epochs(deployed, stages[0]["train"], stages[0]["val"], target, device, settings,
                           policy.balance_loss_weight, settings["initial_epochs"], seed)
    deployed.freeze_for_deployment()
    deployed.eval()
    reference = fit_descriptor(deployed, stages[0]["train"][0], device, batch_size)
    records = [{"stage": 1, "action": "INITIAL_TRAIN", "experts": deployed.num_experts,
                "current": evaluate(deployed, stages[0]["val"], target, device, batch_size),
                "historic": None, "test": None,
                "training": initial, "replay_rows": 0, "replay_identities": [],
                "deployed_state_sha256": state_digest(deployed)}]
    replay_mode = continual["replay_mode"]
    if replay_mode not in {"none", "buffer", "full"} or ((replay_mode == "buffer") != (continual["replay_size"] > 0)):
        raise ValueError("Invalid replay mode/capacity")
    buffer = ReplayBuffer(continual["replay_size"] if replay_mode == "buffer" else 0, seed + 71)
    buffer.add(stages[0]["train"], 1, metadata["stages"][0]["train_source_ids"])
    test_digest = hashlib.sha256(",".join(map(str, metadata["test_source_ids"])).encode()).hexdigest()

    def publish_stage(number):
        """Observe an already committed deployment; never feed test results to selection."""
        record = records[-1]
        record["test"] = evaluate(
            deployed, test, target, device, batch_size,
            prediction_path=output_dir / f"predictions-stage-{number:03d}.csv",
            source_ids=metadata["test_source_ids"], window_centers=metadata.get("test_window_center_s"))
        record["test_source_identity_sha256"] = test_digest
        record["validation_by_stage"] = [
            {"stage": i + 1, **evaluate(deployed, item["val"], target, device, batch_size)}
            for i, item in enumerate(stages[:number])
            if i == number - 1 or continual["history_validation"]]
        record["new_training_rows"] = len(stages[number - 1]["train"][0])
        record["cumulative_training_rows"] = sum(len(item["train"][0]) for item in stages[:number])
        record["cumulative_after_training_rows"] = sum(
            len(item["train"][0]) for item, membership in zip(stages[:number], metadata["stages"][:number])
            if membership.get("source_side") == "after")
        record["checkpoint"] = f"checkpoint-stage-{number:03d}.pt"
        for name in (record["checkpoint"], "checkpoint.pt"):
            save_checkpoint(output_dir / name, deployed, target=target, runtime_config=config, metadata=metadata,
                            descriptor_reference=reference, controller_records=records, stage=number)
        print(f"Stage {number}: fixed test RMSE={record['test']['rmse_km']:.6f} km; reporting only", flush=True)

    publish_stage(1)
    for index, stage in enumerate(stages[1:], 2):
        print(f"Stage {index}: candidate evaluation from {deployed.num_experts} deployed experts", flush=True)
        before_digest = state_digest(deployed)
        prefixes = ["stem."]
        frozen_digest = state_digest(deployed, prefixes)
        current_before = evaluate(deployed, stage["val"], target, device, batch_size)
        historic_data = merge_datasets([item["val"] for item in stages[:index - 1]]) if continual["history_validation"] else None
        old_before = evaluate(deployed, historic_data, target, device, batch_size) if historic_data is not None else None
        novelty = novelty_score(deployed, stage["train"][0], reference, device, batch_size)
        if replay_mode == "full":
            replay = merge_datasets([item["train"] for item in stages[:index - 1]])
            replay_identities = [{"stage": number, "source_id": source_id}
                                 for number, membership in enumerate(metadata["stages"][:index - 1], 1)
                                 for source_id in membership["train_source_ids"]]
        else:
            replay = buffer.dataset()
            replay_identities = deepcopy(buffer.identities)
        training_data = merge_datasets([stage["train"], replay]) if replay is not None else stage["train"]
        adapt_model = deepcopy(deployed)
        adapt_training = train_epochs(adapt_model, training_data, stage["val"], target, device, settings,
                                       policy.balance_loss_weight, settings["adapt_epochs"], seed + index * 100, phase="adapt")
        def candidate_report(candidate, history):
            if state_digest(candidate, prefixes) != frozen_digest:
                raise RuntimeError("Frozen shared frontend parameters changed")
            current = evaluate(candidate, stage["val"], target, device, batch_size)
            old = evaluate(candidate, historic_data, target, device, batch_size) if historic_data is not None else None
            return {"experts": candidate.num_experts, "current": current, "historic": old, "training": history,
                    "assessment": assess_candidate(policy, current_before, current, old_before, old), "shared_frontend_unchanged": True,
                    "old_expert_updates": expert_updates(deployed, candidate)}
        adapted = candidate_report(adapt_model, adapt_training)
        probe, probe_reason = propose_add(policy, novelty, adapted["assessment"], deployed.num_experts, continual["max_experts"])
        added, added_model = None, None
        if probe:
            torch.manual_seed(seed + index * 100 + 1)
            torch.cuda.manual_seed_all(seed + index * 100 + 1)
            added_model = deepcopy(deployed)
            new_id = added_model.add_expert(continual["max_experts"])
            warmup = train_epochs(added_model, stage["train"], stage["val"], target, device, settings, 0.0,
                                  settings["warmup_epochs"], seed + index * 100 + 2, phase="warmup", new_expert_id=new_id)
            joint = train_epochs(added_model, training_data, stage["val"], target, device, settings,
                                 policy.balance_loss_weight, settings["adapt_epochs"], seed + index * 100, phase="adapt", new_expert_id=new_id)
            added = candidate_report(added_model, {"warmup": warmup, "joint": joint, "new_expert_id": new_id})
        action, reason = select_candidate(policy, adapted, added)
        if state_digest(deployed) != before_digest:
            raise RuntimeError("Candidate probing mutated the deployed model")
        if action == "ADAPT_EXISTING":
            deployed = adapt_model
        elif action == "ADD_EXPERT":
            deployed = added_model
        deployed.freeze_for_deployment()
        deployed.eval()
        after_digest = state_digest(deployed)
        if action == "REUSE" and after_digest != before_digest:
            raise RuntimeError("Rejected candidate changed deployment")
        record = {"stage": index, "action": action, "reason": reason, "experts": deployed.num_experts,
                  "novelty": novelty, "novelty_threshold": policy.novelty_threshold,
                  "add_probe": {"executed": probe, "reason": probe_reason},
                  "current_before": current_before, "historic_before": old_before,
                  "adapt_candidate": adapted, "add_candidate": added,
                  "current": evaluate(deployed, stage["val"], target, device, batch_size),
                  "historic": evaluate(deployed, historic_data, target, device, batch_size) if historic_data is not None else None,
                  "test": None,
                  "replay_rows": 0 if replay is None else len(replay[0]), "replay_identities": replay_identities,
                  "old_validation_used_for_training": False,
                  "deployed_before_sha256": before_digest, "deployed_state_sha256": after_digest,
                  "rejected_deployment_unchanged": before_digest == after_digest if action == "REUSE" else None}
        records.append(record)
        buffer.add(stage["train"], index, metadata["stages"][index - 1]["train_source_ids"])
        publish_stage(index)
        print(f"Stage {index}: {action}; current RMSE={record['current']['rmse_km']:.6f} km", flush=True)
    final_test = records[-1]["test"]
    return {"stages": records, "final_test": final_test, "target": target, "descriptor_reference": reference,
            "replay_mode": replay_mode, "historic_validation_access": continual["history_validation"],
            "dormancy": "NOT_IMPLEMENTED", "wakeup": "NOT_IMPLEMENTED", "pruning": "NOT_IMPLEMENTED",
            "threshold_calibration": "uncalibrated", "metadata": metadata,
            "adaptation_policy": "old experts slow, new expert fast, separate router rate, fixed shared frontend",
            "test_evaluation_policy": "fixed holdout after each deployed decision; never used for selection"}
