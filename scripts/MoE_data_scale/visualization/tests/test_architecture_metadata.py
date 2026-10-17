"""Synthetic artifact checks; no numerical runtime or data access."""
from copy import deepcopy
import json
import unittest

from scripts.MoE_data_scale.localization_networks.common.scaling import (
    DEFAULT_SCALING_POLICY, resolve_capacity, serialized_resolution_metadata,
)
from scripts.MoE_data_scale.visualization.results import read_architecture, read_history


class ArchitectureMetadataTests(unittest.TestCase):
    def setUp(self):
        model_config = {
            "input_elements": 21, "input_freq_bins": 8, "base_channels": 4, "dropout": 0.15,
            **resolve_capacity(2, train_count=5, base_channels=4,
                               scaling_policy=DEFAULT_SCALING_POLICY),
        }
        model_config = json.loads(json.dumps(model_config))
        scaling = serialized_resolution_metadata(model_config, "input_elements")
        self.architecture = {
            "pack": "MoE_data_scale", "method": "RBD", "model": "complex_moe_e2_range",
            "model_config": model_config, "scaling": scaling, "balance_loss_weight": 0.01,
            "sampling": {"version": "temporal-block-prefix-v1", "train_fraction": 0.5,
                         "candidate_train_count": 10, "train_count": 5, "validation_count": 3,
                         "block_rows": 32, "seed": 2026, "source_identity_sha256": "a" * 64},
        }
        self.config = {
            "training": {"base_channels": 4, "train_fraction": 0.5, "batch_size": 2, "loss_space": "km"},
            "scaling": dict(DEFAULT_SCALING_POLICY), "moe": {"balance_loss_weight": 0.01},
            "validation": {"seed": 2026},
        }
        self.job = {"method": "RBD", "model": "complex_moe_e2_range", "architecture": "complex_MoE_e2"}
        self.report_row = {"scaling": scaling, "train_fraction": 0.5}
        row = {"epoch": 1, "loss_space": "km", "lr": 0.001, "optimizer_steps_epoch": 3,
               "optimizer_steps": 3, "epoch_elapsed_s": 0.1}
        for split, count in (("train", 5), ("val", 3)):
            row.update({f"{split}_loss": 1.0, f"{split}_total_loss": 1.001,
                        f"{split}_balance_loss": 0.1, f"{split}_router_entropy": 0.6,
                        f"{split}_expert_load": [count, count], f"{split}_expert_importance": [0.4, 0.6],
                        f"{split}_mae_km": 0.2, f"{split}_rmse_km": 0.3})
        self.history = [row]

    def read(self, history):
        return read_history(json.dumps(history).encode(), 1, architecture=self.architecture, config=self.config)

    def test_metadata_arrays_and_default_dropout(self):
        self.assertIsInstance(self.architecture["scaling"]["stem_strides"], list)
        self.assertEqual(read_architecture(json.dumps(self.architecture).encode(), self.report_row,
                                           self.job, self.config), self.architecture)
        changed = deepcopy(self.architecture)
        changed["model_config"]["dropout"] = 0.25
        with self.assertRaisesRegex(ValueError, "dropout"):
            read_architecture(json.dumps(changed).encode(), self.report_row, self.job, self.config)

    def test_original_scm_sidecar_remains_readable_without_joint_metadata(self):
        architecture = deepcopy(self.architecture)
        architecture.update(method="SCM", model="scm_complex_moe_e2_range")
        architecture["model_config"]["input_pairs"] = architecture["model_config"].pop("input_elements")
        job = {**self.job, "method": "SCM", "model": architecture["model"]}
        self.assertEqual(read_architecture(json.dumps(architecture).encode(), self.report_row, job, self.config), architecture)

    def test_complete_history(self):
        self.assertEqual(self.read(self.history), self.history)

    def test_reject_routing_and_loss_corruption(self):
        for key, value in (("train_expert_load", [4, 5]), ("val_expert_load", [3]),
                           ("train_expert_importance", [0.4, 0.7]),
                           ("val_expert_importance", [float("nan"), 0.6]),
                           ("train_router_entropy", 1.0), ("val_total_loss", 1.1),
                           ("optimizer_steps", 4), ("optimizer_steps_epoch", True)):
            with self.subTest(key=key):
                changed = deepcopy(self.history)
                changed[0][key] = value
                with self.assertRaises(ValueError):
                    self.read(changed)
        changed = deepcopy(self.history)
        del changed[0]["train_balance_loss"]
        with self.assertRaises(ValueError):
            self.read(changed)


if __name__ == "__main__":
    unittest.main()
