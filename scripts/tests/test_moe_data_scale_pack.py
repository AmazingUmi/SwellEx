"""Stdlib-only public routing and identity for the expert residual MoE pack."""
from __future__ import annotations

import io
from importlib import import_module
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts.packs import MOE_NETWORKS, get_pack
from scripts.run import PROJECT, main
from scripts.run_identity import make_run_id, parse_run_id, validate_run_identity
from scripts.MoE_data_scale.localization_networks.workflow.tasks import model_plan, resolve_selection

PACK = "MoE_data_scale"


class MoEPublicTests(unittest.TestCase):
    def setUp(self):
        self.remote = import_module(get_pack(PACK).dispatcher)
        self.config = self.remote.load_config(self.remote.DEFAULT_CONFIG)

    def test_full_and_smoke_plans(self):
        self.assertEqual(len(model_plan("experiment")), 20)
        self.assertEqual(len(model_plan("smoke")), 8)
        self.assertEqual({row["architecture"] for row in model_plan("smoke")},
                         {"complex_MoE_e1", "complex_MoE_e4"})
        for method in ("RBD", "ELM", "SCM", "RBD_SCM"):
            prefix = "" if method == "RBD" else method.lower() + "_"
            self.assertEqual(resolve_selection(method, "complex_MoE")["model"], prefix + "complex_moe_range")
            for experts in (1, 2, 4, 8):
                self.assertEqual(resolve_selection(method, f"complex_MoE_e{experts}")["model"],
                                 prefix + f"complex_moe_e{experts}_range")

    def test_selected_dispatch(self):
        for network in MOE_NETWORKS:
            with self.subTest(network=network), patch.object(self.remote, "dispatch", return_value=0) as dispatch:
                self.assertEqual(main([PACK, "--preset", "train_before", "--method", "SCM", "--network", network]), 0)
            config, suite, no_cache = dispatch.call_args.args
            self.assertEqual(config["dataset"]["train_side"], "before")
            self.assertEqual(suite, "selected")
            self.assertFalse(no_cache)
            self.assertEqual(model_plan(suite, dispatch.call_args.kwargs["selection"]),
                             [resolve_selection("SCM", network)])

    def test_quick_routes_without_training(self):
        with patch.object(self.remote, "dispatch", return_value=0) as dispatch:
            self.assertEqual(main([PACK, "validate", "--suite", "quick"]), 0)
        self.assertEqual(dispatch.call_args.args[1], "quick")
        self.assertEqual(model_plan("quick"), [])

    def test_foreign_network_fusion_and_matrix_commands_fail(self):
        cases = (["--network", "GRNN"], ["--network", "ResNet18"],
                 ["--network", "complex_CNN"], ["--network", "complex_MoE_e3"],
                 ["--fusion", "raw_concat"],
                 ["local-grnn"], ["study"], ["budget-study"], ["--dry-run"])
        for args in cases:
            with self.subTest(args=args), patch.object(self.remote, "dispatch") as dispatch, \
                    patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                main([PACK, *args])
            dispatch.assert_not_called()
        for other in ("base_pack", "joint_feature_pack", "data_scale_pack"):
            remote = import_module(get_pack(other).dispatcher)
            with self.subTest(pack=other), patch.object(remote, "dispatch") as dispatch, \
                    patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                main([other, "--network", "complex_MoE"])
            dispatch.assert_not_called()

    def test_run_identity_preserves_method_network_and_pack(self):
        for network in MOE_NETWORKS:
            selection = {"method": "ELM", "network": network}
            run_id = make_run_id(PACK, self.config, selection)
            fields = parse_run_id(run_id, PACK)
            self.assertEqual((fields["pack"], fields["method"], fields["network"]), ("moe", "ELM", network))
            validate_run_identity(run_id, PACK, self.config, selection)
            for other in ("base_pack", "joint_feature_pack", "data_scale_pack"):
                with self.subTest(other=other), self.assertRaises(ValueError):
                    parse_run_id(run_id, other)

    def test_bad_recovery_ids_fail_before_ssh(self):
        for run_id in ("study-one", "../unsafe", "20261016T120000Z-scale-full-SCM-all-12345678",
                       "20261016T120000Z-moe-full-SCM-GRNN-12345678",
                       "20261016T120000Z-moe-full-SCM-all-block_rms-12345678"):
            with self.subTest(run_id=run_id), patch.object(self.remote, "remote_python") as ssh, self.assertRaises(ValueError):
                self.remote.status(self.config, run_id)
            ssh.assert_not_called()

    def test_help_and_plan_need_no_numerical_packages(self):
        code = '''import importlib.abc,sys
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy','scipy','h5py','matplotlib'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Reject())
from scripts.MoE_data_scale.localization_networks.workflow.tasks import model_plan
assert len(model_plan('experiment'))==20
from scripts.run import main
main(['MoE_data_scale','--help'])
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("complex_MoE_e8", result.stdout)
        self.assertIn("study", result.stdout)


if __name__ == "__main__":
    unittest.main()
