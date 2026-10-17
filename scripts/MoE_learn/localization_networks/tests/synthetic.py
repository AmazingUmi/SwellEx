"""Synthetic, provenance-complete fixtures; never accesses original observations."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import torch


def synthetic_config():
    from scripts.MoE_learn.localization_networks.workflow.config import load_config
    config = load_config(Path(__file__).resolve().parents[1] / "workflow" / "haiqin_demo.toml")
    config["training"].update(initial_epochs=1, adapt_epochs=1, warmup_epochs=1, batch_size=4,
                              base_channels=2, learning_rate=0.002, old_expert_learning_rate=0.0001,
                              new_expert_learning_rate=0.001, router_learning_rate=0.001,
                              seed=2026, weight_decay=0.0)
    config["continual"].update(stages=2, initial_experts=2, max_experts=3, replay_mode="buffer", replay_size=4, history_validation=True)
    config["controller"].update(min_relative_gain=0.01, max_old_relative_regression=0.05, novelty_threshold=1.0, balance_loss_weight=0.01)
    config["validation"].update(device="cuda", threads=2, seed=2026)
    return config


def synthetic_data(seed=2026, stages=2):
    generator = torch.Generator().manual_seed(seed)
    features, frequencies = 252, 22
    stages_out, identities = [], []
    next_id = 1
    for stage in range(1, stages + 1):
        x = torch.randn(12, 2, features, frequencies, generator=generator)
        if stage > 1:
            x[:, :, :21, :11] += 0.75 * (stage - 1)
        x = (x - x.mean(dim=(2, 3), keepdim=True)) / x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        y = torch.linspace(0.5, 2.5, 12).unsqueeze(1) + 0.15 * (stage - 1)
        stages_out.append({"train_x": x[:8], "train_y": y[:8], "val_x": x[8:], "val_y": y[8:]})
        identities.append({"stage": stage, "train_source_ids": list(range(next_id, next_id + 8)),
                           "val_source_ids": list(range(next_id + 8, next_id + 12))})
        next_id += 12
    test_x = torch.randn(6, 2, features, frequencies, generator=generator)
    test_x = (test_x - test_x.mean(dim=(2, 3), keepdim=True)) / test_x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    test_y = torch.linspace(0.5, 2.5, 6).unsqueeze(1)
    shared = {"frequency_hz": [float(50 + 5 * i) for i in range(frequencies)],
              "element_idx": list(range(1, 22)), "depth_m": [float(100 + i) for i in range(21)]}
    rbd = {"method": "RBD", "input_shape": [2, 21, frequencies], **deepcopy(shared), "pair_indices": []}
    scm = {"method": "SCM", "input_shape": [2, 231, frequencies], **deepcopy(shared),
           "pair_indices": [[i, j] for j in range(1, 22) for i in range(1, j + 1)]}
    signature = {"format_version": 1, "method": "RBD_SCM", "input_shape": [2, features, frequencies],
                 "fusion_strategy": "block_rms", "blocks": [
                     {"method": "RBD", "start": 0, "stop": 21, "input_signature": rbd},
                     {"method": "SCM", "start": 21, "stop": 252, "input_signature": scm}],
                 "source_contract": {"input_files": [{"path": "synthetic://seed-" + str(seed), "size_bytes": 1,
                         "mtime_ns": 0, "sha256": hashlib.sha256(str(seed).encode()).hexdigest()}],
                     "effective_config": {"dataset": {"num_snapshots_per_segment": 1, "snapshot_overlap_count": 0}},
                     "split": {"kind": "synthetic_disjoint"}, "generator_version": "moe-learn-synthetic-v1"}}
    metadata = {"pack": "MoE_learn", "method": "RBD_SCM", "fusion": "block_rms", "input_signature": signature,
                "input_shape": [2, features, frequencies], "source_paths": ["synthetic://seed-" + str(seed)],
                "preprocessing": {"strategy": "block_rms", "scales": [1.0, 1.0], "fit_sample_count": 8,
                    "fit_source_identity_sha256": hashlib.sha256(",".join(map(str, identities[0]["train_source_ids"])).encode()).hexdigest()},
                "preprocessing_fit_stage": 1, "input_normalized_per_sample": True, "stages": identities,
                "test_source_ids": list(range(next_id, next_id + 6)),
                "evidence_scope": "synthetic only; no SwellEx accuracy or environment-generalization evidence"}
    return {"stages": stages_out, "test_x": test_x, "test_y": test_y, "metadata": metadata}
