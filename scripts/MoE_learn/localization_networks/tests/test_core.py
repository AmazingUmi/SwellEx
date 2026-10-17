"""Critical incremental-training contracts. Never skip an unavailable GPU."""
from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.MoE_learn.localization_networks.common.checkpoints import load_checkpoint, save_checkpoint, state_digest, validate_metadata
from scripts.MoE_learn.localization_networks.common.controller import ControllerPolicy, assess_candidate, propose_add, select_candidate
from scripts.MoE_learn.localization_networks.common.model import ContinualMoE, ModelConfig
from scripts.MoE_learn.localization_networks.common.training import ReplayBuffer, optimizer_groups, require_cuda, run_stages, train_epochs
from .synthetic import synthetic_config, synthetic_data


class ContinualContracts(unittest.TestCase):
    def setUp(self):
        self.device = require_cuda(os.environ.get("SWELLEX_MOE_LEARN_TEST_DEVICE", "cuda"))
        torch.manual_seed(2026)

    def test_default_shapes_and_sparse_gradients(self):
        model = ContinualMoE(ModelConfig(252, 22, 16, 4, 0.0)).to(self.device)
        x = torch.randn(2, 2, 252, 22, device=self.device)
        shapes = []
        hooks = [model.stem[0].register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape))),
                 model.stem[1].register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape)))]
        prediction, aux = model.forward_with_aux(x)
        for hook in hooks:
            hook.remove()
        self.assertEqual(shapes, [(2, 32, 252, 11), (2, 64, 126, 6)])
        self.assertEqual(tuple(prediction.shape), (2, 1))
        self.assertEqual(tuple(model.descriptor(x).shape), (2, 64))
        (prediction.square().mean() + 0.01 * aux["balance_loss"]).backward()
        selected = set(aux["selected_experts"].flatten().tolist())
        self.assertEqual(int(aux["expert_load"].sum()), 4)
        for index, expert in enumerate(model.experts):
            gradients = [p.grad for p in expert.parameters()]
            self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients) if index in selected else all(g is None for g in gradients))

    def test_expansion_warmup_freezes_old_function_and_preserves_router_rows(self):
        model = ContinualMoE(ModelConfig(252, 22, 2, 2, 0.5)).to(self.device).eval()
        x = torch.randn(2, 2, 252, 22, device=self.device)
        old_prediction = model.expert_prediction(x, 0).detach().clone()
        old_weight, old_bias = model.router.weight.detach().clone(), model.router.bias.detach().clone()
        prefixes = ["stem.", "experts.0.", "experts.1."]
        before = state_digest(model, prefixes)
        self.assertEqual(model.add_expert(4), 2)
        self.assertTrue(torch.equal(model.router.weight[:2], old_weight))
        self.assertTrue(torch.equal(model.router.bias[:2], old_bias))
        model.train()
        self.assertFalse(model.experts[0].training)
        self.assertTrue(model.experts[2].training)
        self.assertFalse(model.stem.training)
        self.assertTrue(all(not p.requires_grad for p in model.router.parameters()))
        self.assertTrue(torch.equal(old_prediction, model.expert_prediction(x, 0)))
        settings = synthetic_config()["training"]
        groups, _ = optimizer_groups(model, settings, phase="warmup", new_expert_id=2)
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
        router_before = state_digest(model, ["router."])
        prediction = model.expert_prediction(x, 2)
        optimizer.zero_grad(set_to_none=True)
        prediction.square().mean().backward()
        optimizer.step()
        self.assertEqual(before, state_digest(model, prefixes))
        self.assertEqual(router_before, state_digest(model, ["router."]))
        self.assertTrue(all(p.grad is None for p in model.stem.parameters()))
        self.assertTrue(all(p.grad is None for p in model.experts[0].parameters()))

    def _check_group_contract(self, model, groups, metadata):
        self.assertEqual(len(groups), len(metadata))
        actual = [parameter for group in groups for parameter in group["params"]]
        self.assertEqual(len(actual), len({id(parameter) for parameter in actual}))
        self.assertEqual({id(parameter) for parameter in actual},
                         {id(parameter) for parameter in model.parameters() if parameter.requires_grad})
        for group, record in zip(groups, metadata):
            self.assertEqual(group["name"], record["role"])
            self.assertEqual(group["lr"], record["learning_rate"])
            self.assertEqual(sum(parameter.numel() for parameter in group["params"]), record["parameter_count"])
            self.assertTrue(group["params"])
            self.assertIsInstance(record["expert_ids"], list)

    def _check_adaptation_groups(self, model, groups, metadata, settings, new_expert_id=None):
        self._check_group_contract(model, groups, metadata)
        old_ids = [index for index in range(model.num_experts) if index != new_expert_id]
        expected = [({id(p) for index in old_ids for p in model.experts[index].parameters()},
                     settings["old_expert_learning_rate"], old_ids),
                    ({id(p) for p in model.router.parameters()}, settings["router_learning_rate"], [])]
        if new_expert_id is not None:
            expected.append(({id(p) for p in model.experts[new_expert_id].parameters()},
                             settings["new_expert_learning_rate"], [new_expert_id]))
        self.assertEqual(len(groups), len(expected))
        for members, learning_rate, expert_ids in expected:
            matches = [(group, record) for group, record in zip(groups, metadata)
                       if {id(parameter) for parameter in group["params"]} == members]
            self.assertEqual(len(matches), 1)
            group, record = matches[0]
            self.assertEqual(group["lr"], learning_rate)
            self.assertEqual(record["expert_ids"], expert_ids)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.stem.parameters()))

    def test_optimizer_groups_cover_each_trainable_parameter_once_at_correct_rates(self):
        model = ContinualMoE(ModelConfig(252, 22, 2, 2, 0.0)).to(self.device)
        settings = synthetic_config()["training"]
        groups, metadata = optimizer_groups(model, settings, phase="initial")
        self._check_group_contract(model, groups, metadata)
        self.assertEqual({id(p) for group in groups for p in group["params"]}, {id(p) for p in model.parameters()})
        self.assertTrue(all(group["lr"] == settings["learning_rate"] for group in groups))
        new_id = model.add_expert(3)
        with self.assertRaises(ValueError):
            optimizer_groups(model, settings, phase="warmup")
        groups, metadata = optimizer_groups(model, settings, phase="warmup", new_expert_id=new_id)
        self._check_group_contract(model, groups, metadata)
        self.assertEqual(len(groups), 1)
        self.assertEqual({id(p) for p in groups[0]["params"]}, {id(p) for p in model.experts[new_id].parameters()})
        self.assertEqual(groups[0]["lr"], settings["new_expert_learning_rate"])
        self.assertEqual(metadata[0]["expert_ids"], [new_id])
        groups, metadata = optimizer_groups(model, settings, phase="adapt", new_expert_id=new_id)
        self._check_adaptation_groups(model, groups, metadata, settings, new_id)

    def test_joint_adaptation_updates_old_experts_but_not_shared_frontend(self):
        model = ContinualMoE(ModelConfig(252, 22, 2, 2, 0.0)).to(self.device)
        model.freeze_for_deployment()
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        settings = synthetic_config()["training"]
        groups, metadata = optimizer_groups(model, settings, phase="adapt")
        self._check_adaptation_groups(model, groups, metadata, settings)
        model.train()
        self.assertFalse(model.stem.training)
        self.assertTrue(all(expert.training for expert in model.experts))
        old_before = [state_digest(model, [f"experts.{index}."]) for index in range(2)]
        stem_before = state_digest(model, ["stem."])
        x = torch.randn(4, 2, 252, 22, device=self.device)
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        prediction, aux = model.forward_with_aux(x)
        target = torch.linspace(-0.5, 0.5, len(x), device=self.device).unsqueeze(1)
        ((prediction - target).square().mean() + 0.01 * aux["balance_loss"]).backward()
        for expert in model.experts:
            gradients = [parameter.grad for parameter in expert.parameters()]
            self.assertTrue(all(value is not None and bool(torch.isfinite(value).all()) for value in gradients))
            self.assertTrue(any(bool(value.abs().sum() > 0) for value in gradients))
        self.assertTrue(all(p.grad is None for p in model.stem.parameters()))
        optimizer.step()
        self.assertEqual(stem_before, state_digest(model, ["stem."]))
        for index, before in enumerate(old_before):
            self.assertNotEqual(before, state_digest(model, [f"experts.{index}."]))

    def test_old_learning_rate_scales_identical_gradient_updates_and_new_expert_ages(self):
        model = ContinualMoE(ModelConfig(252, 22, 2, 2, 0.0)).to(self.device)
        new_id = model.add_expert(3)
        settings = synthetic_config()["training"]
        groups, metadata = optimizer_groups(model, settings, phase="adapt", new_expert_id=new_id)
        self._check_adaptation_groups(model, groups, metadata, settings, new_id)
        old_bias = model.experts[0].head[-1].bias
        new_bias = model.experts[new_id].head[-1].bias
        with torch.no_grad():
            old_bias.zero_()
            new_bias.zero_()
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        old_bias.grad = torch.ones_like(old_bias)
        new_bias.grad = torch.ones_like(new_bias)
        optimizer.step()
        old_delta, new_delta = float(old_bias.detach().abs().item()), float(new_bias.detach().abs().item())
        self.assertGreater(old_delta, 0.0)
        self.assertGreater(new_delta, old_delta)
        self.assertAlmostEqual(old_delta / new_delta,
                               settings["old_expert_learning_rate"] / settings["new_expert_learning_rate"], places=6)
        model.freeze_for_deployment()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        groups, metadata = optimizer_groups(model, settings, phase="adapt")
        self._check_adaptation_groups(model, groups, metadata, settings)
        former_new_group = next(group for group in groups if any(parameter is new_bias for parameter in group["params"]))
        self.assertEqual(former_new_group["lr"], settings["old_expert_learning_rate"])

    def test_new_expert_warmup_cannot_starve(self):
        model = ContinualMoE(ModelConfig(252, 22, 2, 2, 0.0)).to(self.device)
        new_id = model.add_expert(3)
        with torch.no_grad():
            model.router.weight[new_id].zero_()
            model.router.bias[new_id].fill_(-100)
        prepared, config = synthetic_data(), synthetic_config()
        stage = prepared["stages"][0]
        x, y = stage["train_x"], stage["train_y"]
        before = state_digest(model, [f"experts.{new_id}."])
        target = {"mean_km": 1.0, "std_km": 1.0}
        frozen_before = state_digest(model, ["stem.", "experts.0.", "experts.1.", "router."])
        history = train_epochs(model, (x, y), (x, y), target, self.device, config["training"], 0.0, 1, 2026,
                               phase="warmup", new_expert_id=new_id)
        self.assertNotEqual(before, state_digest(model, [f"experts.{new_id}."]))
        self.assertEqual(frozen_before, state_digest(model, ["stem.", "experts.0.", "experts.1.", "router."]))
        self.assertEqual(history["history"][0]["expert_load"][new_id], len(x))

    def test_controller_guard_rejection_and_novelty_gate(self):
        policy = ControllerPolicy(min_relative_gain=0.05, max_old_relative_regression=0.05)
        metric = lambda value: {"rmse_km": value}
        bad = assess_candidate(policy, metric(1), metric(0.8), metric(1), metric(1.2))
        self.assertFalse(bad["accepted"])
        self.assertEqual(bad["historic_guard_status"], "FAIL")
        good = assess_candidate(policy, metric(1), metric(0.8), metric(1), metric(1.01))
        self.assertFalse(propose_add(policy, 0.1, good, 2, 4)[0])
        self.assertTrue(propose_add(policy, 2.0, good, 2, 4)[0])
        self.assertTrue(propose_add(policy, 0.1, bad, 2, 4)[0])
        self.assertFalse(propose_add(policy, 2.0, bad, 4, 4)[0])
        self.assertEqual(select_candidate(policy, {"assessment": bad})[0], "REUSE")
        close = {"assessment": good, "current": metric(0.79)}
        self.assertEqual(select_candidate(policy, {"assessment": good, "current": metric(0.8)}, close)[0], "ADAPT_EXISTING")

    def test_replay_contains_only_supplied_optimizer_members(self):
        prepared = synthetic_data()
        buffer = ReplayBuffer(4, 2026)
        stage = prepared["stages"][0]
        ids = prepared["metadata"]["stages"][0]["train_source_ids"]
        buffer.add((stage["train_x"], stage["train_y"]), 1, ids)
        self.assertEqual(len(buffer.rows), 4)
        self.assertTrue({row["source_id"] for row in buffer.identities}.issubset(ids))
        no_replay = ReplayBuffer(0, 2026)
        no_replay.add((stage["train_x"], stage["train_y"]), 1, ids)
        self.assertIsNone(no_replay.dataset())
        self.assertEqual(no_replay.identities, [])

    def test_stages_reject_without_mutating_deployment_and_restore(self):
        prepared, config = synthetic_data(), synthetic_config()
        config["controller"]["min_relative_gain"] = 1.0
        config["continual"].update(replay_mode="none", replay_size=0, history_validation=False)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = run_stages(config, prepared, output, device=self.device)
            record = result["stages"][1]
            self.assertEqual(record["action"], "REUSE")
            self.assertEqual(record["deployed_before_sha256"], record["deployed_state_sha256"])
            self.assertEqual(record["replay_rows"], 0)
            self.assertIsNone(record["historic"])
            self.assertEqual(record["adapt_candidate"]["assessment"]["historic_guard_status"], "NOT_RUN")
            self.assertTrue(record["adapt_candidate"]["shared_frontend_unchanged"])
            updates = record["adapt_candidate"]["old_expert_updates"]
            self.assertEqual([item["expert_id"] for item in updates], [0, 1])
            self.assertTrue(all(item["changed"] and item["l2_delta"] > 0 and math.isfinite(item["l2_delta"]) for item in updates))
            self.assertEqual(record["adapt_candidate"]["training"]["phase"], "adapt")
            self.assertIsNotNone(record["add_candidate"])
            self.assertTrue(record["add_candidate"]["shared_frontend_unchanged"])
            self.assertTrue(any(item["changed"] and item["l2_delta"] > 0 for item in record["add_candidate"]["old_expert_updates"]))
            restored, payload = load_checkpoint(output / "checkpoint.pt", self.device, expected_metadata=prepared["metadata"])
            self.assertEqual(state_digest(restored), record["deployed_state_sha256"])
            self.assertEqual(payload["target"], result["target"])
            self.assertEqual(payload["stage"], 2)
            self.assertFalse(restored.training)
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in restored.parameters()))
            reference = restored(prepared["test_x"].to(self.device)).detach()
            second, _ = load_checkpoint(output / "checkpoint.pt", self.device)
            self.assertTrue(torch.equal(reference, second(prepared["test_x"].to(self.device))))
            restored.configure_training(range(restored.num_experts), train_router=True)
            restored.train()
            self.assertTrue(all(parameter.requires_grad for expert in restored.experts for parameter in expert.parameters()))
            self.assertTrue(all(parameter.requires_grad for parameter in restored.router.parameters()))
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in restored.stem.parameters()))
            self.assertFalse(restored.stem.training)
            foreign = deepcopy(payload)
            foreign["pack"] = "MoE_data_scale"
            torch.save(foreign, output / "foreign.pt")
            with self.assertRaises(ValueError):
                load_checkpoint(output / "foreign.pt", self.device)
            incompatible = deepcopy(prepared["metadata"])
            for block in incompatible["input_signature"]["blocks"]:
                block["input_signature"]["frequency_hz"][0] += 1.0
            with self.assertRaises(ValueError):
                load_checkpoint(output / "checkpoint.pt", self.device, expected_metadata=incompatible)
            changed_scale = deepcopy(prepared["metadata"])
            changed_scale["preprocessing"]["scales"][0] = 2.0
            with self.assertRaises(ValueError):
                load_checkpoint(output / "checkpoint.pt", self.device, expected_metadata=changed_scale)
            for field in ("stages", "test_source_ids"):
                changed_members = deepcopy(prepared["metadata"])
                if field == "stages":
                    changed_members["stages"][1]["val_source_ids"][0] += 1000
                else:
                    changed_members["test_source_ids"][0] += 1000
                with self.subTest(membership=field), self.assertRaisesRegex(ValueError, f"Restored {field}"):
                    load_checkpoint(output / "checkpoint.pt", self.device, expected_metadata=changed_members)

            # Reuse the deployment to test artifact identity, without another training run.
            holdout_config = deepcopy(payload["runtime_config"])
            holdout_config["dataset"].update(stage_mode="distance_holdout", split_strategy="range_overlap",
                                              train_side="before", distance_edges_km=[.5, 1.5, 2.5], heldout_group=2)
            holdout_metadata = deepcopy(payload["metadata"])
            first, second_stage = holdout_metadata["stages"]
            holdout_metadata["stage_mode"] = "distance_holdout"
            holdout_metadata["allocation"] = {
                "heldout_group": 2, "distance_edges_km": [.5, 1.5, 2.5],
                "distance_groups": [
                    {"group": 1, "before_source_ids": first["train_source_ids"] + first["val_source_ids"],
                     "after_source_ids": second_stage["train_source_ids"] + second_stage["val_source_ids"]},
                    {"group": 2, "before_source_ids": [1000], "after_source_ids": list(holdout_metadata["test_source_ids"])}],
                "before": {"excluded_source_ids": [1000]},
                "after": {"test_source_ids": list(holdout_metadata["test_source_ids"])},
            }
            restored.freeze_for_deployment()

            def save_identity(path, runtime, metadata):
                return save_checkpoint(path, restored, target=payload["target"], runtime_config=runtime,
                    metadata=metadata, descriptor_reference=payload["descriptor_reference"],
                    controller_records=payload["controller_records"], stage=payload["stage"])

            holdout_path = output / "holdout.pt"
            holdout_payload = save_identity(holdout_path, holdout_config, holdout_metadata)
            load_checkpoint(holdout_path, self.device, expected_metadata=holdout_metadata)
            for field, value in (("heldout_group", 1), ("distance_edges_km", [.25, 1.5, 2.5])):
                wrong_config = deepcopy(holdout_config)
                wrong_config["dataset"][field] = value
                with self.subTest(save_mismatch=field), self.assertRaisesRegex(ValueError, field):
                    save_identity(output / "invalid-save.pt", wrong_config, holdout_metadata)
                tampered = deepcopy(holdout_payload)
                tampered["runtime_config"] = wrong_config
                torch.save(tampered, output / "tampered.pt")
                with self.subTest(load_mismatch=field), self.assertRaisesRegex(ValueError, field):
                    load_checkpoint(output / "tampered.pt", self.device)
            for mode in (None, "interleaved"):
                wrong_metadata = deepcopy(holdout_metadata)
                if mode is None:
                    del wrong_metadata["stage_mode"]
                else:
                    wrong_metadata["stage_mode"] = mode
                with self.subTest(save_mode=mode), self.assertRaisesRegex(ValueError, "stage_mode"):
                    save_identity(output / "invalid-save.pt", holdout_config, wrong_metadata)
                tampered = deepcopy(holdout_payload)
                tampered["metadata"] = wrong_metadata
                torch.save(tampered, output / "tampered.pt")
                with self.subTest(load_mode=mode), self.assertRaisesRegex(ValueError, "stage_mode"):
                    load_checkpoint(output / "tampered.pt", self.device)
            with self.assertRaisesRegex(ValueError, "stage_mode"):
                save_identity(output / "invalid-save.pt", payload["runtime_config"], holdout_metadata)
            for membership in ("distance_groups", "excluded_before"):
                different_fold = deepcopy(holdout_metadata)
                if membership == "distance_groups":
                    different_fold["allocation"]["distance_groups"][0]["after_source_ids"][0] += 1000
                else:
                    different_fold["allocation"]["before"]["excluded_source_ids"][0] += 1000
                with self.subTest(fold_membership=membership), self.assertRaisesRegex(ValueError, "Restored allocation"):
                    load_checkpoint(holdout_path, self.device, expected_metadata=different_fold)

    def test_metadata_rejects_preprocessing_leakage(self):
        prepared = synthetic_data()
        metadata = deepcopy(prepared["metadata"])
        metadata["preprocessing"]["fit_sample_count"] += 1
        with self.assertRaises(ValueError):
            validate_metadata(metadata, ModelConfig(252, 22, 2, 2))

    def test_full_replay_and_stage_decisions_do_not_depend_on_test_labels(self):
        prepared, config = synthetic_data(stages=4), synthetic_config()
        config["continual"].update(stages=4, replay_mode="full", replay_size=0)
        changed = deepcopy(prepared)
        changed["test_y"] = changed["test_y"] + 10.0
        results = []
        for fixture in (prepared, changed):
            with tempfile.TemporaryDirectory() as directory:
                result = run_stages(config, fixture, Path(directory), device=self.device)
                results.append(result)
                for number, record in enumerate(result["stages"], 1):
                    expected = [{"stage": index, "source_id": source_id}
                                for index, membership in enumerate(fixture["metadata"]["stages"][:number - 1], 1)
                                for source_id in membership["train_source_ids"]]
                    self.assertEqual(record["replay_identities"], expected)
                    self.assertEqual(record["replay_rows"], len(expected))
                    self.assertEqual(record["cumulative_training_rows"], number * 8)
                    self.assertEqual(record["test"]["count"], 6)
                    self.assertTrue((Path(directory) / f"predictions-stage-{number:03d}.csv").is_file())
                    restored, payload = load_checkpoint(Path(directory) / record["checkpoint"], self.device)
                    self.assertEqual(payload["stage"], number)
                    self.assertEqual(state_digest(restored), record["deployed_state_sha256"])
        for before, after in zip(results[0]["stages"], results[1]["stages"]):
            self.assertEqual(before["action"], after["action"])
            self.assertEqual(before["deployed_state_sha256"], after["deployed_state_sha256"])
            self.assertEqual(before["test_source_identity_sha256"], after["test_source_identity_sha256"])
            self.assertNotEqual(before["test"]["rmse_km"], after["test"]["rmse_km"])

    def test_window_centering_ignores_other_windows_and_preserves_raw_inputs(self):
        import numpy as np
        from scripts.MoE_learn.data_processing.generation_core import centered_window, load_inputs
        signal = np.arange(60, dtype=np.float64).reshape(3, 20)
        original = signal.copy()
        expected = centered_window(signal, 5, 10)
        np.testing.assert_array_equal(signal, original)
        signal[:, :5] += 10000
        signal[:, 10:] -= 10000
        np.testing.assert_array_equal(expected, centered_window(signal, 5, 10))
        np.testing.assert_allclose(expected.mean(axis=1), 0)
        # Guard the input-loading boundary too: no global centering before extraction.
        positions = np.array([[1., 100.], [2., 120.]])
        ranges = np.array([[0., 1.], [1., 2.]])
        ctd = np.array([[0., 0., 0., 1500., 0.], [200., 0., 0., 1500., 0.]])
        channels = [np.arange(20, dtype=np.float64) + offset for offset in (100, 200)]
        with patch("scripts.MoE_learn.data_processing.generation_core.np.loadtxt", side_effect=[positions, ranges, ctd]), \
                patch("scripts.MoE_learn.data_processing.generation_core.load_channel", side_effect=channels), \
                patch("scripts.MoE_learn.data_processing.generation_core._input_file_identity", return_value={}):
            loaded = load_inputs(Path("/synthetic"), {"dataset": {"event_name": "S5"}})
        np.testing.assert_array_equal(loaded["signal"], np.stack(channels))


if __name__ == "__main__":
    unittest.main()
