"""Remote-only numerical and restoration tests for the scalable CNN family."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import unittest

import torch

from scripts.data_scale_pack.localization_networks.common.scaling import DEFAULT_SCALING_POLICY, PROFILE_REPEATS


def registry(method):
    return importlib.import_module(f"scripts.data_scale_pack.localization_networks.{method}.network.model")


def config_for(module, name, rows=8, **overrides):
    kwargs = {"base_channels": 2, "dropout": 0.0, "train_count": 600, "scaling_policy": DEFAULT_SCALING_POLICY}
    kwargs.update(overrides)
    return module.build_model_config(name, (2, rows, 22), **kwargs)


class ScalableCNNTests(unittest.TestCase):
    def test_calibrated_auto_is_runtime_identical_to_virtual_fixed_observation(self):
        artifact_path = Path(__file__).resolve().parents[2] / "workflow/scm_capacity_calibration_v2.json"
        artifact = json.loads(artifact_path.read_text())
        self.assertEqual(artifact["policy_id"], "validation-calibrated-v2")
        self.assertEqual(artifact["method"], "SCM")
        self.assertEqual(artifact["base_channels"], 16)
        self.assertNotEqual(artifact["policy"], DEFAULT_SCALING_POLICY)
        module = registry("scm")
        auto_name = "scm_complex_cnn_range"
        expected_profiles = {337: "small", 675: "small", 1350: "medium", 2700: "large"}
        with torch.random.fork_rng(devices=[]):
            for train_count, expected_profile in expected_profiles.items():
                with self.subTest(train_count=train_count, profile=expected_profile):
                    torch.manual_seed(2029)
                    before_resolution = torch.random.get_rng_state().clone()
                    auto_config = config_for(module, auto_name, rows=231, base_channels=16,
                                             dropout=0.15, train_count=train_count, scaling_policy=artifact["policy"])
                    self.assertEqual(auto_config.selected_profile, expected_profile)
                    self.assertEqual(auto_config.selection_mode, "auto")
                    self.assertEqual(auto_config.scaling_policy, artifact["policy"])
                    self.assertTrue(torch.equal(before_resolution, torch.random.get_rng_state()))
                    fixed_name = f"scm_complex_cnn_{expected_profile}_range"
                    fixed_config = config_for(module, fixed_name, rows=231, base_channels=16,
                                              dropout=0.15, train_count=train_count, scaling_policy=DEFAULT_SCALING_POLICY)
                    self.assertEqual(fixed_config.selection_mode, "fixed")
                    self.assertEqual(fixed_config.scaling_policy, DEFAULT_SCALING_POLICY)
                    auto = module.build_model(auto_name, auto_config).eval()
                    torch.manual_seed(2029)
                    fixed = module.build_model(fixed_name, fixed_config).eval()
                    self.assertIs(type(auto), type(fixed))
                    self.assertEqual(auto_config.parameter_count, fixed_config.parameter_count)
                    self.assertEqual(sum(p.numel() for p in auto.parameters()), auto_config.parameter_count)
                    self.assertEqual(sum(p.numel() for p in fixed.parameters()), fixed_config.parameter_count)
                    auto_state, fixed_state = auto.state_dict(), fixed.state_dict()
                    self.assertEqual(list(auto_state), list(fixed_state))
                    for key in auto_state:
                        self.assertTrue(torch.equal(auto_state[key], fixed_state[key]), key)
                    x = torch.randn(2, 2, 231, 22)
                    with torch.no_grad():
                        torch.testing.assert_close(auto(x), fixed(x), rtol=0, atol=0)

    def test_every_method_and_profile_forward_backward_and_parameter_count(self):
        for method, rows in (("rbd", 8), ("elm", 28), ("scm", 36)):
            module = registry(method)
            self.assertEqual(len(module.MODEL_REGISTRY), 6)
            for name in module.MODEL_REGISTRY:
                with self.subTest(method=method, model=name):
                    config = config_for(module, name, rows)
                    model = module.build_model(name, config)
                    self.assertEqual(sum(p.numel() for p in model.parameters()), config.parameter_count)
                    output = model(torch.randn(2, 2, rows, 22))
                    self.assertEqual(tuple(output.shape), (2, 1))
                    output.square().mean().backward()
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_depth_does_not_add_downsampling(self):
        module = registry("rbd")
        for profile, repeats in PROFILE_REPEATS.items():
            name = f"complex_cnn_{profile}_range"
            model = module.build_model(name, config_for(module, name))
            strides = [block.block[0].real.stride for block in model.features]
            self.assertEqual(len(strides), sum(repeats))
            self.assertEqual([stride for stride in strides if stride != (1, 1)], [(1, 2), (2, 2), (2, 1)])
            features = model.features(torch.randn(2, 2, 8, 22))
            self.assertEqual(tuple(features.shape[-2:]), (2, 6))

    def test_baseline_matches_original_weights_and_outputs(self):
        # The preserved base pack is an independent original-architecture oracle.
        from scripts.base_pack.localization_networks.rbd.network.models.complex_cnn import (
            ComplexRangeCNN as Original,
            ModelConfig as OriginalConfig,
        )
        module = registry("rbd")
        config = config_for(module, "complex_cnn_baseline_range")
        baseline = module.build_model("complex_cnn_baseline_range", config).eval()
        original = Original(OriginalConfig(input_elements=8, input_freq_bins=22, base_channels=2, dropout=0.0)).eval()
        original.load_state_dict(baseline.state_dict(), strict=True)
        x = torch.randn(2, 2, 8, 22)
        torch.testing.assert_close(baseline(x), original(x), rtol=0, atol=0)

    def test_checkpoint_roundtrip_uses_saved_layout_for_every_model(self):
        for method in ("rbd", "elm", "scm"):
            module = registry(method)
            for name in module.MODEL_REGISTRY:
                with self.subTest(method=method, model=name):
                    config = config_for(module, name, train_count=4000)
                    saved = {"model_name": name, "model_config": module.serialize_model_config(config)}
                    # JSON roundtrips tuples as arrays; reconstruction canonicalizes them.
                    restored_name, restored_config = module.model_config_from_checkpoint(json.loads(json.dumps(saved)))
                    self.assertEqual(restored_name, name)
                    self.assertEqual(restored_config, config)
                    self.assertEqual(module.resolution_metadata(restored_config)["calibration_status"], "uncalibrated")
                    original = module.build_model(name, config).eval()
                    restored = module.build_model(name, restored_config).eval()
                    restored.load_state_dict(original.state_dict(), strict=True)
                    x = torch.randn(2, 2, 8, 22)
                    torch.testing.assert_close(original(x), restored(x), rtol=0, atol=0)

    def test_reject_incomplete_and_incoherent_checkpoints(self):
        module = registry("rbd")
        config = config_for(module, "complex_cnn_range")
        saved = {"model_name": "complex_cnn_range", "model_config": module.serialize_model_config(config)}
        for key in saved["model_config"]:
            broken = deepcopy(saved)
            del broken["model_config"][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                module.model_config_from_checkpoint(broken)
        for key, value in (
            ("train_count", 1), ("parameter_count", 1), ("parameter_budget", 1.0),
            ("scaling_policy_version", "old"), ("stage_repeats", (1, 1, 1)),
            ("stage_strides", ((1, 1), (1, 1), (1, 1))), ("selected_profile", "small"),
            ("selection_mode", "fixed"), ("dropout", 1.0), ("input_elements", 0),
        ):
            broken = deepcopy(saved)
            broken["model_config"][key] = value
            with self.subTest(changed=key), self.assertRaises(ValueError):
                module.model_config_from_checkpoint(broken)
        broken = deepcopy(saved)
        broken["model_name"] = "complex_cnn_small_range"
        with self.assertRaises(ValueError):
            module.model_config_from_checkpoint(broken)

    def test_build_rejects_unknown_missing_or_wrong_shape_arguments(self):
        module = registry("rbd")
        with self.assertRaises(ValueError):
            module.build_model_config("complex_cnn_range", (2, 8, 22))
        with self.assertRaises(ValueError):
            config_for(module, "complex_cnn_range", obsolete_option=True)
        for shape in ((1, 8, 22), (True, 8, 22), (2, 0, 22), (2, 8, 0), (2, 8)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                module.build_model_config("complex_cnn_range", shape, train_count=600, scaling_policy=DEFAULT_SCALING_POLICY)


if __name__ == "__main__":
    unittest.main()
