"""Public routing and task identity for the data-size scaling experiment."""
from __future__ import annotations

import copy
import io
from importlib import import_module
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts.packs import get_pack
from scripts.run import PROJECT, main
from scripts.run_identity import make_run_id, parse_run_id, validate_run_identity
from scripts.data_scale_pack.localization_networks.workflow.tasks import model_plan, resolve_selection

PACK = "data_scale_pack"
NETWORKS = (
    "complex_CNN", "complex_CNN_small", "complex_CNN_medium",
    "complex_CNN_large", "complex_CNN_xlarge", "complex_CNN_baseline",
)


class DataScalePublicTests(unittest.TestCase):
    def setUp(self):
        self.remote = import_module(get_pack(PACK).dispatcher)

    def test_plan_contains_auto_four_fixed_profiles_and_original_baseline(self):
        full = model_plan("experiment")
        smoke = model_plan("smoke")
        self.assertEqual(len(full), 18)
        self.assertEqual(len(smoke), 6)
        self.assertEqual({row["architecture"] for row in full}, set(NETWORKS))
        self.assertEqual({row["architecture"] for row in smoke}, {"complex_CNN", "complex_CNN_baseline"})
        self.assertEqual({row["method"] for row in full}, {"RBD", "ELM", "SCM"})
        self.assertEqual(len({(row["method"], row["model"]) for row in full}), 18)
        for method in ("RBD", "ELM", "SCM"):
            prefix = "" if method == "RBD" else method.lower() + "_"
            self.assertEqual(resolve_selection(method, "complex_CNN")["model"], prefix + "complex_cnn_range")
            self.assertEqual(resolve_selection(method, "complex_CNN_small")["model"], prefix + "complex_cnn_small_range")

    def test_each_public_profile_dispatches_only_the_selected_model(self):
        for network in NETWORKS:
            args = [PACK, "--preset", "train_before", "--method", "RBD", "--network", network]
            with self.subTest(network=network), patch.object(self.remote, "dispatch", return_value=0) as dispatch:
                self.assertEqual(main(args), 0)
            config, suite, no_cache = dispatch.call_args.args
            self.assertEqual(suite, "selected")
            self.assertEqual(config["dataset"]["train_side"], "before")
            self.assertFalse(no_cache)
            selection = dispatch.call_args.kwargs["selection"]
            self.assertEqual(model_plan(suite, selection), [resolve_selection("RBD", network)])

    def test_foreign_networks_and_fusion_are_rejected_without_launch(self):
        for args in (["--network", "GRNN"], ["--network", "ResNet18"],
                     ["--network", "ResNet50"], ["--network", "real_CNN"],
                     ["--network", "complex_CNN_auto"], ["--network", "complex_cnn_small"],
                     ["--method", "RBD_SCM"], ["--fusion", "raw_concat"], ["local-grnn"]):
            with self.subTest(args=args), patch.object(self.remote, "dispatch") as dispatch, \
                    patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                main([PACK, *args])
            dispatch.assert_not_called()
        for pack_name in ("base_pack", "joint_feature_pack"):
            remote = import_module(get_pack(pack_name).dispatcher)
            for network in NETWORKS[1:]:
                with self.subTest(pack=pack_name, network=network), patch.object(remote, "dispatch") as dispatch, \
                        patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                    main([pack_name, "--network", network])
                dispatch.assert_not_called()

    def test_profile_run_id_preserves_pack_split_and_selection(self):
        config = self.remote.load_config(self.remote.DEFAULT_CONFIG)
        config["dataset"].update(split_strategy="range_overlap", train_side="after")
        for network in NETWORKS:
            selection = {"method": "SCM", "network": network}
            run_id = make_run_id(PACK, config, selection)
            fields = parse_run_id(run_id, PACK)
            self.assertEqual((fields["pack"], fields["split"], fields["method"], fields["network"]),
                             ("scale", "after", "SCM", network))
            validate_run_identity(run_id, PACK, config, selection)
            changed = copy.deepcopy(config)
            changed["dataset"]["train_side"] = "before"
            with self.assertRaises(ValueError):
                validate_run_identity(run_id, PACK, changed, selection)
            with self.assertRaises(ValueError):
                parse_run_id(run_id, "base_pack")
            with self.assertRaises(ValueError):
                parse_run_id(run_id, "joint_feature_pack")

    def test_foreign_architectures_in_task_ids_fail_before_ssh(self):
        invalid = (
            "20261015T120000Z-scale-full-all-GRNN-12345678",
            "20261015T120000Z-scale-full-all-ResNet18-12345678",
            "20261015T120000Z-scale-full-all-complex_CNN_auto-12345678",
            "20261015T120000Z-scale-full-RBD_SCM-complex_CNN-12345678",
            "20261015T120000Z-scale-full-all-all-block_rms-12345678",
            "20261015T120000Z-base-full-all-complex_CNN_small-12345678",
            "20261015T120000Z-joint-full-all-complex_CNN_small-block_rms-12345678",
        )
        config = self.remote.load_config(self.remote.DEFAULT_CONFIG)
        for run_id in invalid:
            with self.subTest(run_id=run_id), patch.object(self.remote, "remote_python") as ssh, self.assertRaises(ValueError):
                self.remote.status(config, run_id)
            ssh.assert_not_called()

    def test_help_and_plan_imports_do_not_load_numerical_dependencies(self):
        code = """import importlib.abc,sys
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy','scipy','h5py','matplotlib'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Reject())
from scripts.data_scale_pack.localization_networks.workflow.tasks import model_plan
from scripts.data_scale_pack.localization_networks.workflow.study_remote import is_study_id
from scripts.data_scale_pack.visualization.results import load_run
assert len(model_plan('experiment')) == 18
assert is_study_id('scm-capacity-budget-study-v2')
assert not is_study_id('20261015T120000Z-scale-full-SCM-all-12345678')
from scripts.run import main
main(['data_scale_pack','--help'])
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("complex_CNN_baseline", result.stdout)
        self.assertNotIn("GRNN", result.stdout)

    def test_budget_study_routes_plan_and_rejects_experiment_options(self):
        from pathlib import Path
        module = import_module("scripts.data_scale_pack.localization_networks.workflow.study_remote")
        with patch.object(module, "run_study", return_value=0) as run:
            self.assertEqual(main([PACK, "budget-study", "--plan", "plan.json", "--dry-run"]), 0)
        run.assert_called_once_with(Path("plan.json"), study_dir=None, dry_run=True, kind="budget")
        with patch.object(module, "run_study", return_value=0) as run:
            self.assertEqual(main([PACK, "study", "--plan", "plan.json",
                                   "--study-dir", "output/data/my-study"]), 0)
        run.assert_called_once_with(Path("plan.json"), study_dir=Path("output/data/my-study"),
                                    dry_run=False, kind="scale")
        for args in (["budget-study"], ["--dry-run"],
                     ["study", "--plan", "plan.json", "--dry-run"],
                     ["budget-study", "--plan", "plan.json", "--method", "SCM"],
                     ["budget-study", "--plan", "plan.json", "--config", "other.toml"]):
            with self.subTest(args=args), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                main([PACK, *args])

    def test_study_status_and_fetch_use_unified_entry_and_custom_remote_config(self):
        module = import_module("scripts.data_scale_pack.localization_networks.workflow.study_remote")
        study_id = "scm-capacity-budget-study-v2"
        config = self.remote.load_config(self.remote.DEFAULT_CONFIG)
        for command in ("status", "fetch"):
            for state, exit_code in (("PASS", 0), ("RUNNING", 2), ("FAIL", 1)):
                report = {"run_id": study_id, "study_id": study_id, "status": state}
                with self.subTest(command=command, state=state), \
                        patch.object(module, command, return_value=report) as action, \
                        patch.object(module, "summarize") as summary, \
                        patch.object(self.remote, command) as child_action, \
                        patch.object(self.remote, "load_config", return_value=config) as load:
                    self.assertEqual(main([PACK, command, "--run", study_id,
                                           "--config", "custom.toml"]), exit_code)
                action.assert_called_once_with(config, study_id)
                child_action.assert_not_called()
                summary.assert_called_once_with(report)
                self.assertEqual(load.call_args.args[0].name, "custom.toml")

    def test_child_status_and_fetch_do_not_route_through_study(self):
        module = import_module("scripts.data_scale_pack.localization_networks.workflow.study_remote")
        run_id = "20261015T120000Z-scale-full-SCM-all-12345678"
        for command in ("status", "fetch"):
            report = {"run_id": run_id, "status": "PASS", "checks": []}
            with self.subTest(command=command), \
                    patch.object(self.remote, command, return_value=report) as child_action, \
                    patch.object(self.remote, "summarize") as summary, \
                    patch.object(module, command) as study_action:
                self.assertEqual(main([PACK, command, "--run", run_id]), 0)
            self.assertEqual(child_action.call_args.args[1], run_id)
            study_action.assert_not_called()
            summary.assert_called_once_with(report)

    def test_default_generated_study_id_is_not_mistaken_for_a_child_run(self):
        module = import_module("scripts.data_scale_pack.localization_networks.workflow.study_remote")
        study_id = "20261016T070009Z-study-1234abcd"
        self.assertTrue(module.is_study_id(study_id))
        for command in ("status", "fetch"):
            report = {"run_id": study_id, "study_id": study_id, "status": "RUNNING"}
            with self.subTest(command=command), \
                    patch.object(module, command, return_value=report) as action, \
                    patch.object(module, "summarize"), \
                    patch.object(self.remote, command) as child_action:
                self.assertEqual(main([PACK, command, "--run", study_id]), 2)
            self.assertEqual(action.call_args.args[1], study_id)
            child_action.assert_not_called()
        for invalid in ("20261316T070009Z-study-1234abcd",
                        "20261016T070009Z-study-short"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                module.is_study_id(invalid)

    def test_invalid_or_cross_pack_child_ids_cannot_fall_back_to_study_lookup(self):
        module = import_module("scripts.data_scale_pack.localization_networks.workflow.study_remote")
        invalid = (
            "20261015T120000Z-base-full-SCM-all-12345678",
            "20261015T120000Z-joint-full-SCM-all-block_rms-12345678",
            "20261015T120000Z-scale-full-SCM-ResNet18-12345678",
            "20261015T120000Z-scale-full-SCM-all-bad",
            "../scm-capacity-budget-study-v2",
        )
        for command in ("status", "fetch"):
            for run_id in invalid:
                with self.subTest(command=command, run_id=run_id), \
                        patch.object(module, command) as study_action, \
                        patch.object(self.remote, "remote_python") as ssh, \
                        patch("sys.stderr", io.StringIO()):
                    self.assertEqual(main([PACK, command, "--run", run_id]), 1)
                study_action.assert_not_called()
                ssh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
