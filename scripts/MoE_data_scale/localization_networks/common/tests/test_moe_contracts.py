"""Synthetic CUDA checks: architecture correctness, not scientific performance.

Run through the pack's remote ``validate --suite quick``. Direct unittest runs
must explicitly set SWELLEX_MOE_TEST_DEVICE=cuda (or cuda:N); CPU is rejected.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import io
import math
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.MoE_data_scale.localization_networks.common import scaling
from scripts.MoE_data_scale.localization_networks.common.checkpoints import CHECKPOINT_VERSION, validate_checkpoint
from scripts.MoE_data_scale.localization_networks.common.input_contract import require_matching_input_signature
from scripts.MoE_data_scale.localization_networks.common.training_utils import run_epoch

REQUESTED_DEVICE = None
PACKAGE = "scripts.MoE_data_scale.localization_networks"


def registry(method):
    return importlib.import_module(f"{PACKAGE}.{method.lower()}.network.model")


def make_model(method="RBD", experts=4, *, channels=2, rows=None, frequencies=22, train_count=600):
    from scripts.MoE_data_scale.localization_networks.workflow.tasks import resolve_selection
    public = "complex_MoE" if experts == "auto" else f"complex_MoE_e{experts}"
    name = resolve_selection(method, public)["model"]
    module = registry(method)
    shape = (2, rows or {"RBD": 21, "ELM": 210, "SCM": 231}[method], frequencies)
    config = module.build_model_config(name, shape, base_channels=channels, dropout=0.0,
                                       train_count=train_count, scaling_policy=dict(scaling.DEFAULT_SCALING_POLICY))
    return module.build_model(name, config), config, name


def signature(method, rows, frequencies):
    elements = 21
    pairs = [] if method == "RBD" else [[i, j] for j in range(1, elements + 1)
                                                for i in range(1, j + (method == "SCM"))]
    return {"method": method, "input_shape": [2, rows, frequencies],
            "frequency_hz": [float(100 + i * 5) for i in range(frequencies)],
            "element_idx": list(range(1, elements + 1)), "depth_m": [100.0 + i for i in range(elements)],
            "pair_indices": pairs}


class MoEContracts(unittest.TestCase):
    def setUp(self):
        requested = REQUESTED_DEVICE or os.environ.get("SWELLEX_MOE_TEST_DEVICE")
        if not requested:
            raise RuntimeError("Set SWELLEX_MOE_TEST_DEVICE explicitly; CUDA numerical checks have no CPU fallback")
        self.device = torch.device(requested)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Residual MoE numerical contracts require available CUDA; CPU fallback forbidden")
        self.assertTrue(torch.ones(1, device=self.device).is_cuda)
        torch.manual_seed(2026)

    def model(self, *args, **kwargs):
        model, config, name = make_model(*args, **kwargs)
        return model.to(self.device), config, name

    def assert_finite_gradient(self, tensor):
        self.assertIsNotNone(tensor.grad)
        self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
        self.assertGreater(float(tensor.grad.abs().sum()), 0.0)

    def test_residual_is_exact_identity_and_shortcut_has_gradient(self):
        model, _, _ = self.model()
        block = model.experts[0].residual
        for parameter in block.branch.parameters():
            nn.init.zeros_(parameter)
        x = torch.randn(1, 16, 3, 2, device=self.device, requires_grad=True)
        y = block(x)
        self.assertTrue(torch.equal(x, y), "Zero F must yield exact x+F(x), including negative values")
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)), "Identity shortcut derivative must be one")
        self.assertFalse(any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()))
        self.assertTrue(any(isinstance(module, nn.GroupNorm) for module in model.modules()))

    def test_sparse_dispatch_weights_and_router_only_mixes_outputs(self):
        model, config, _ = self.model(experts=8)
        model.eval()
        with torch.no_grad():
            model.router.weight.zero_()
            model.router.bias.copy_(torch.tensor([3., 2., -5., -6., -7., -8., -9., -10.], device=self.device))
        captures = {}
        handles = []
        for index, expert in enumerate(model.experts):
            def capture(module, args, output, e=index):
                captures[e] = (args[0].shape[0], output)
            handles.append(expert.register_forward_hook(capture))
        x = torch.randn(3, 2, 21, 22, device=self.device, requires_grad=True)
        pred, aux = model.forward_with_aux(x)
        for handle in handles:
            handle.remove()
        self.assertEqual(set(captures), {0, 1}, "Unselected experts must not execute")
        self.assertEqual([captures[e][0] for e in (0, 1)], [3, 3])
        self.assertEqual(tuple(aux["selected_experts"].shape), (3, 2))
        self.assertTrue(bool((aux["selected_experts"] == torch.tensor([0, 1], device=self.device)).all()))
        weights = aux["routing_weights"]
        self.assertTrue(bool((weights >= 0).all()))
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(3, device=self.device)))
        self.assertEqual(int(aux["expert_load"].sum()), 6)
        self.assertEqual(aux["expert_load"].tolist(), [3, 3, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(float(aux["expert_importance"].sum()), 3.0, places=5)
        self.assertTrue(torch.allclose(pred, captures[0][1] * weights[:, :1] + captures[1][1] * weights[:, 1:2]))
        (pred.square().mean() + aux["balance_loss"]).backward()
        self.assert_finite_gradient(x)
        self.assert_finite_gradient(model.router.bias)
        for index, expert in enumerate(model.experts):
            if index > 1:
                self.assertTrue(all(parameter.grad is None for parameter in expert.parameters()))
        self.assertEqual(config.top_k, 2)

    def test_expert_copies_are_independent_and_growth_preserves_layout(self):
        layouts = []
        counts = []
        for experts in (1, 2, 4, 8):
            model, config, _ = self.model(experts=experts)
            layouts.append([(name, tuple(parameter.shape)) for name, parameter in model.experts[0].named_parameters()])
            counts.append(sum(parameter.numel() for parameter in model.parameters()))
            self.assertEqual(counts[-1], scaling.parameter_count(experts, 2))
            self.assertEqual(config.active_parameter_count, scaling.active_parameter_count(experts, 2))
            self.assertEqual(len(model.experts), experts)
            self.assertEqual(model.router is None, experts == 1)
            if model.router is not None:
                self.assertEqual((model.router.in_features, model.router.out_features), (4, experts))
            pointers = [parameter.data_ptr() for expert in model.experts for parameter in expert.parameters()]
            self.assertEqual(len(pointers), len(set(pointers)))
            if experts > 1:
                first = next(model.experts[0].parameters())
                other = next(model.experts[1].parameters())
                before = other.detach().clone()
                with torch.no_grad():
                    first.add_(1)
                self.assertTrue(torch.equal(before, other))
        self.assertTrue(all(layout == layouts[0] for layout in layouts))
        self.assertEqual(counts, sorted(set(counts)))

    def test_singleton_and_e1_e2_edges_are_finite(self):
        for experts in (1, 2, 4, 8):
            with self.subTest(experts=experts):
                model, _, _ = self.model(experts=experts, rows=1, frequencies=1)
                x = torch.randn(1, 2, 1, 1, device=self.device, requires_grad=True)
                if model.router is not None:
                    with torch.no_grad():
                        model.router.weight.zero_()
                        model.router.bias.copy_(torch.arange(experts, dtype=torch.float32, device=self.device))
                pred, aux = model.forward_with_aux(x)
                self.assertTrue(bool(torch.isfinite(pred).all()))
                self.assertEqual(tuple(pred.shape), (1, 1))
                self.assertEqual(int(aux["expert_load"].sum()), min(2, experts))
                self.assertTrue(torch.allclose(aux["routing_weights"].sum(1), torch.ones(1, device=self.device)))
                self.assertTrue(bool(torch.isfinite(aux["router_entropy"])))
                self.assertTrue(aux["balance_loss"].requires_grad)
                if experts == 1:
                    self.assertEqual(float(aux["balance_loss"].detach()), 0.0)
                    self.assertEqual(aux["routing_weights"].tolist(), [[1.0]])
                else:
                    gradient = torch.autograd.grad(aux["balance_loss"], model.router.bias, retain_graph=True)[0]
                    self.assertTrue(bool(torch.isfinite(gradient).all()))
                    self.assertGreater(float(gradient.abs().sum()), 0.0)
                (pred.sum() + aux["balance_loss"]).backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_strict_configuration_and_invalid_counts(self):
        module = registry("RBD")
        _, config, name = self.model(experts="auto")
        payload = module.serialize_model_config(config)
        expected = module.resolution_metadata(config)
        self.assertEqual(expected["calibration_status"], "uncalibrated")
        self.assertEqual(config.num_experts, 2)
        for count in (False, 0, -1, 1.5, float("nan")):
            with self.subTest(count=count), self.assertRaises(ValueError):
                module.build_model_config(name, (2, 21, 22), train_count=count,
                                          scaling_policy=dict(scaling.DEFAULT_SCALING_POLICY))
        for kwargs in ({"base_channels": 1}, {"base_channels": True}, {"dropout": 1.0},
                       {"dropout": float("nan")}, {"unknown": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                module.build_model_config(name, (2, 21, 22), train_count=600,
                                          scaling_policy=dict(scaling.DEFAULT_SCALING_POLICY), **kwargs)
        for field, value in (("num_experts", 3), ("top_k", 1), ("parameter_count", 1),
                             ("active_parameter_count", 1), ("architecture_version", "old"),
                             ("residual_blocks", 2), ("train_count", 1), ("normalization", "batchnorm")):
            bad = deepcopy(payload)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                scaling.validate_serialized_config(bad, "input_elements")
        missing = dict(payload)
        del missing["num_experts"]
        with self.assertRaises(ValueError):
            scaling.validate_serialized_config(missing, "input_elements")
        with self.assertRaises(ValueError):
            scaling.validate_scaling_policy({"reference_samples": 600, "exponent": .5})
        for shape in ((1, 21, 22), (2, 0, 22), (2, 21), (2, 21, True)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                module.build_model_config(name, shape, train_count=600, scaling_policy=dict(scaling.DEFAULT_SCALING_POLICY))
        model, _, _ = self.model()
        for x in (torch.zeros(0, 2, 21, 22, device=self.device), torch.zeros(1, 1, 21, 22, device=self.device),
                  torch.full((1, 2, 21, 22), float("nan"), device=self.device)):
            with self.assertRaises(ValueError):
                model(x)

    def test_checkpoint_roundtrip_freezes_experts_and_rejects_identity_changes(self):
        for method in ("RBD", "ELM", "SCM"):
            with self.subTest(method=method):
                model, config, name = self.model(method, experts="auto")
                module = registry(method)
                shape = (2, {"RBD": 21, "ELM": 210, "SCM": 231}[method], 22)
                axes = signature(method, shape[1], shape[2])
                checkpoint = {"checkpoint_version": CHECKPOINT_VERSION, "pack": "MoE_data_scale",
                              "method": method, "model_name": name, "dataset_name": "synthetic-contract",
                              "model_config": module.serialize_model_config(config), "scaling": module.resolution_metadata(config),
                              "sampling": {"train_count": 600, "train_fraction": 1.0}, "train_fraction": 1.0,
                              "source_identity": {"synthetic": True}, "model_state": model.state_dict(),
                              "input_signature": axes, "input_normalized_per_sample": True,
                              "target_mean_km": 3.0, "target_std_km": 1.0, "loss_space": "normalized",
                              "huber_beta": 1.0, "balance_loss_weight": .01}
                validate_checkpoint(checkpoint, method=method, model_name=name, dataset_name="synthetic-contract")
                stream = io.BytesIO()
                torch.save(checkpoint, stream)
                stream.seek(0)
                loaded = torch.load(stream, map_location=self.device, weights_only=False)
                validate_checkpoint(loaded, method=method, model_name=name, dataset_name="synthetic-contract")
                with patch.object(module, "build_model_config", side_effect=AssertionError("Prediction must not reselect for new N")):
                    restored_name, restored_config = module.model_config_from_checkpoint(loaded)
                    restored = module.build_model(restored_name, restored_config).to(self.device)
                restored.load_state_dict(loaded["model_state"])
                self.assertEqual(restored_config.num_experts, config.num_experts)
                self.assertEqual(restored_config.train_count, 600)
                changed_n = module.build_model_config(name, shape, base_channels=2, dropout=0.0,
                                                     train_count=9600, scaling_policy=dict(scaling.DEFAULT_SCALING_POLICY))
                self.assertNotEqual(changed_n.num_experts, restored_config.num_experts)
                x = torch.randn(1, *shape, device=self.device)
                model.eval(); restored.eval()
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(x), restored(x)))
                for key, value in (("pack", "data_scale_pack"), ("method", "JOINT"), ("model_name", "obsolete"),
                                   ("balance_loss_weight", -1), ("balance_loss_weight", float("nan"))):
                    bad = deepcopy(checkpoint)
                    bad[key] = value
                    with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                        validate_checkpoint(bad, method=method, model_name=name, dataset_name="synthetic-contract")
                for field in ("balance_loss_weight", "input_signature", "scaling"):
                    bad = dict(checkpoint)
                    del bad[field]
                    with self.assertRaises(ValueError):
                        validate_checkpoint(bad, method=method, model_name=name, dataset_name="synthetic-contract")
                bad = deepcopy(checkpoint)
                del bad["input_signature"]["frequency_hz"]
                with self.assertRaises(ValueError):
                    validate_checkpoint(bad, method=method, model_name=name, dataset_name="synthetic-contract")
                if axes["pair_indices"]:
                    bad = deepcopy(checkpoint)
                    bad["input_signature"]["pair_indices"].reverse()
                    with self.assertRaises(ValueError):
                        validate_checkpoint(bad, method=method, model_name=name, dataset_name="synthetic-contract")
                for axis in ("frequency_hz", "depth_m", "pair_indices"):
                    if not axes[axis]:
                        continue
                    changed = deepcopy(axes)
                    changed[axis].reverse()
                    with self.assertRaises(ValueError):
                        require_matching_input_signature(loaded["input_signature"], changed, "Synthetic prediction")
                require_matching_input_signature(loaded["input_signature"], axes, "Synthetic prediction")

    def test_optimizer_integration_uses_one_aux_forward_and_preserves_regression_metrics(self):
        model, _, _ = self.model(experts=4)
        with torch.no_grad():
            model.router.bias.copy_(torch.tensor([1., .5, -.5, -1.], device=self.device))
        x = torch.randn(4, 2, 21, 22, device=self.device)
        y = torch.tensor([[2.0], [3.0], [4.0], [5.0]], device=self.device)
        loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
        criterion = nn.MSELoss()
        before = model.router.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        with patch.object(model, "forward_with_aux", wraps=model.forward_with_aux) as forward:
            for _ in range(3):
                metrics = run_epoch(model, loader, criterion, self.device, 3., 2., optimizer=optimizer, balance_loss_weight=.2)
                self.assertAlmostEqual(metrics["total_loss"], metrics["loss"] + .2 * metrics["balance_loss"], places=5)
                self.assertTrue(all(math.isfinite(metrics[key]) for key in ("loss", "total_loss", "balance_loss", "rmse_km")))
                self.assertEqual(sum(metrics["expert_load"]), 8)
                self.assertAlmostEqual(sum(metrics["expert_importance"]), 1.0, places=5)
            self.assertEqual(forward.call_count, 6, "One router/expert forward per batch")
        self.assertFalse(torch.equal(before, model.router.weight))
        model.eval()
        with torch.no_grad():
            pred, aux = model.forward_with_aux(x)
            raw_rmse = float(((pred * 2. + 3. - y).square().mean()).sqrt())
            raw_regression = float(criterion(pred, (y - 3.) / 2.))
        full_loader = DataLoader(TensorDataset(x, y), batch_size=4)
        no_aux = run_epoch(model, full_loader, criterion, self.device, 3., 2., balance_loss_weight=0.)
        with_aux = run_epoch(model, full_loader, criterion, self.device, 3., 2., balance_loss_weight=.2)
        self.assertAlmostEqual(with_aux["rmse_km"], raw_rmse, places=5)
        self.assertAlmostEqual(with_aux["loss"], raw_regression, places=5)
        self.assertEqual(no_aux["rmse_km"], with_aux["rmse_km"])
        self.assertGreater(with_aux["total_loss"], no_aux["total_loss"])
        for weight in (True, -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                run_epoch(model, full_loader, criterion, self.device, 3., 2., balance_loss_weight=weight)


if __name__ == "__main__":
    unittest.main()
