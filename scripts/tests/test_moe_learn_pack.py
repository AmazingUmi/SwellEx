"""Public continual-MoE contracts; standard library only, with no SSH or models."""
from __future__ import annotations

import copy
from importlib import import_module
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.packs import PACKS, get_pack
from scripts.run import PROJECT, main
from scripts.run_identity import make_run_id, parse_run_id, validate_run_identity

PACK = "MoE_learn"
METHOD = "RBD_SCM"
NETWORK = "complex_MoE_learn"


class MoELearnPublicTests(unittest.TestCase):
    def setUp(self):
        self.remote = import_module(get_pack(PACK).dispatcher)
        self.config_module = import_module(f"scripts.{PACK}.localization_networks.workflow.config")
        self.config = self.remote.load_config(self.remote.DEFAULT_CONFIG)

    def test_default_and_demo_preset_dispatch_the_same_config(self):
        observed = []
        for args in ([], ["--preset", "demo"]):
            with patch.object(self.remote, "dispatch", return_value=0) as dispatch:
                self.assertEqual(main([PACK, *args]), 0)
            config, suite, no_cache = dispatch.call_args.args
            self.assertEqual(suite, "experiment")
            self.assertFalse(no_cache)
            observed.append(config)
        self.assertEqual(observed[0], observed[1])
        self.assertEqual(get_pack(PACK).methods, (METHOD,))
        self.assertEqual(get_pack(PACK).networks, (NETWORK,))

    def test_explicit_selection_and_validation_suites(self):
        with patch.object(self.remote, "dispatch", return_value=0) as dispatch:
            self.assertEqual(main([PACK, "--method", METHOD, "--network", NETWORK]), 0)
        selection = dispatch.call_args.kwargs["selection"]
        self.assertEqual(selection, {"method": METHOD, "network": NETWORK})
        plan = self.remote.model_plan(dispatch.call_args.args[1], selection)
        self.assertEqual(len(plan), 1)
        self.assertEqual((plan[0]["method"], plan[0]["architecture"]), (METHOD, NETWORK))
        for args, suite in ((["validate"], "smoke"),
                            (["validate", "--suite", "quick"], "quick"),
                            (["validate", "--suite", "experiment"], "experiment")):
            with self.subTest(args=args), patch.object(self.remote, "dispatch", return_value=0) as dispatch:
                self.assertEqual(main([PACK, *args]), 0)
            self.assertEqual(dispatch.call_args.args[1], suite)
        self.assertEqual(self.remote.model_plan("quick"), [])

    def test_dry_run_prints_plan_without_ssh_dispatch_or_source_snapshot(self):
        for args in (["--dry-run"], ["validate", "--suite", "quick", "--dry-run"],
                     ["--preset", "demo", "--method", METHOD, "--network", NETWORK, "--dry-run"]):
            with self.subTest(args=args), patch.object(self.remote, "dispatch") as dispatch, \
                    patch.object(self.remote, "remote_python") as ssh, \
                    patch.object(self.remote, "freeze_source") as snapshot, \
                    patch("sys.stdout", io.StringIO()) as output:
                self.assertEqual(main([PACK, *args]), 0)
            dispatch.assert_not_called()
            ssh.assert_not_called()
            snapshot.assert_not_called()
            result = json.loads(output.getvalue())
            self.assertEqual(result["pack"], PACK)

    def test_invalid_public_arguments_do_not_launch(self):
        invalid = (["--preset", "full"], ["--preset", "train_before"],
                   ["--method", "RBD"], ["--network", "complex_MoE"],
                   ["--network", "GRNN"], ["--fusion", "block_rms"],
                   ["study"], ["local-grnn"], ["--epochs", "1"],
                   ["--method", METHOD, "--method", METHOD],
                   ["--preset", "demo", "--config", "missing.toml"],
                   ["--suite", "quick"], ["status"],
                   ["fetch", "--run", "x", "--preset", "demo"],
                   ["check", "--dry-run"], ["status", "--run", "x", "--dry-run"])
        for args in invalid:
            with self.subTest(args=args), patch.object(self.remote, "dispatch") as dispatch, \
                    patch.object(self.remote, "remote_python") as ssh, \
                    patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                main([PACK, *args])
            dispatch.assert_not_called()
            ssh.assert_not_called()

    def test_identity_preserves_pack_and_selected_method_network(self):
        selection = {"method": METHOD, "network": NETWORK}
        run_id = make_run_id(PACK, self.config, selection)
        fields = parse_run_id(run_id, PACK)
        self.assertEqual(fields["pack"], get_pack(PACK).tag)
        self.assertEqual((fields["method"], fields["network"]), (METHOD, NETWORK))
        validate_run_identity(run_id, PACK, self.config, selection)
        for other in PACKS:
            if other == PACK:
                continue
            with self.subTest(pack=other), self.assertRaises(ValueError):
                parse_run_id(run_id, other)
        with self.assertRaises(ValueError):
            validate_run_identity(run_id, PACK, self.config, None)

    def test_recovery_rejects_foreign_unsafe_and_wrong_report_identities(self):
        other = import_module(get_pack("MoE_data_scale").dispatcher)
        foreign = make_run_id("MoE_data_scale", other.load_config(other.DEFAULT_CONFIG))
        for command in (self.remote.status, self.remote.fetch):
            for run_id in (foreign, "../outside", "study-one", "20261017T120000Z-12345678"):
                with self.subTest(command=command.__name__, run_id=run_id), \
                        patch.object(self.remote, "remote_python") as ssh, self.assertRaises(ValueError):
                    command(self.config, run_id)
                ssh.assert_not_called()
        run_id = make_run_id(PACK, self.config)
        for report in ({"run_id": run_id, "pack": "MoE_data_scale", "status": "PASS"},
                       {"run_id": foreign, "pack": PACK, "status": "PASS"}):
            with patch.object(self.remote, "remote_python", return_value=report), self.assertRaises(ValueError):
                self.remote.status(self.config, run_id)

    def test_config_rejects_missing_unknown_and_legacy_fields(self):
        for table, values in self.config.items():
            if not isinstance(values, dict):
                continue
            for operation in ("unknown", "missing"):
                invalid = copy.deepcopy(self.config)
                if operation == "unknown":
                    invalid[table]["obsolete_alias"] = 1
                else:
                    del invalid[table][next(iter(values))]
                with self.subTest(table=table, operation=operation), self.assertRaises(ValueError):
                    self.config_module.validate_config(invalid)
        for key, value in (("schema_version", True), ("fusion", {"mode": "raw_concat"})):
            invalid = copy.deepcopy(self.config)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.config_module.validate_config(invalid)

    def test_config_rejects_values_that_cannot_reach_the_cuda_model(self):
        cases = (("validation", "device", "cpu"), ("validation", "threads", True),
                 ("training", "base_channels", 1), ("training", "initial_epochs", 0),
                 ("training", "adapt_epochs", -1), ("training", "warmup_epochs", -1),
                 ("training", "batch_size", 1.5), ("training", "learning_rate", float("nan")),
                 ("training", "old_expert_learning_rate", 0),
                 ("training", "new_expert_learning_rate", float("inf")),
                 ("training", "router_learning_rate", -1),
                 ("training", "weight_decay", float("inf")), ("training", "seed", -1),
                 ("dataset", "stage_mode", "random"), ("dataset", "val_fraction", 1),
                 ("dataset", "frequency_selection_modes", ["adapt"]),
                 ("continual", "stages", 1), ("continual", "initial_experts", 1),
                 ("continual", "max_experts", 9), ("continual", "history_validation", 1),
                 ("controller", "min_relative_gain", 1.01),
                 ("controller", "max_old_relative_regression", -0.01),
                 ("controller", "novelty_threshold", 0), ("controller", "balance_loss_weight", True))
        for table, key, value in cases:
            invalid = copy.deepcopy(self.config)
            invalid[table][key] = value
            with self.subTest(table=table, key=key, value=value), self.assertRaises(ValueError):
                self.config_module.validate_config(invalid)

    def test_old_experts_require_a_positive_rate_lower_than_new_experts(self):
        for rate in (self.config["training"]["new_expert_learning_rate"], 1.0):
            invalid = copy.deepcopy(self.config)
            invalid["training"]["old_expert_learning_rate"] = rate
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                self.config_module.validate_config(invalid)

    def test_replay_and_history_validation_have_independent_explicit_contracts(self):
        for history_validation in (False, True):
            valid = copy.deepcopy(self.config)
            valid["continual"].update(replay_mode="none", replay_size=0,
                                     history_validation=history_validation)
            self.config_module.validate_config(valid)
            valid["continual"].update(replay_mode="buffer", replay_size=4)
            self.config_module.validate_config(valid)
            valid["continual"].update(replay_mode="full", replay_size=0)
            self.config_module.validate_config(valid)
        for mode, size in (("none", 1), ("buffer", 0), ("full", 1), ("old", 4)):
            invalid = copy.deepcopy(self.config)
            invalid["continual"].update(replay_mode=mode, replay_size=size)
            with self.subTest(mode=mode, size=size), self.assertRaises(ValueError):
                self.config_module.validate_config(invalid)
        invalid = copy.deepcopy(self.config)
        invalid["continual"].update(initial_experts=4, max_experts=2)
        with self.assertRaises(ValueError):
            self.config_module.validate_config(invalid)

    def test_before_then_after_requires_explicit_holdout_and_before_source(self):
        valid = copy.deepcopy(self.config)
        valid["dataset"].update(stage_mode="before_then_after", split_strategy="range_overlap",
                                train_side="before", after_test_fraction=0.2, after_grouping="time_interleaved")
        valid["continual"].update(stages=4, replay_mode="full", replay_size=0)
        self.config_module.validate_config(valid)
        for key, value in (("train_side", "after"), ("split_strategy", "full_interleaved"),
                           ("after_test_fraction", 0), ("after_test_fraction", 1), ("after_grouping", "random")):
            invalid = copy.deepcopy(valid)
            invalid["dataset"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.config_module.validate_config(invalid)
        for key in ("after_test_fraction", "after_grouping"):
            invalid = copy.deepcopy(valid)
            del invalid["dataset"][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.config_module.validate_config(invalid)
        invalid = copy.deepcopy(self.config)
        invalid["training"]["seed"] = invalid["validation"]["seed"] + 1
        with self.assertRaises(ValueError):
            self.config_module.validate_config(invalid)

    def test_range_groups_require_explicit_positive_finite_increasing_edges_for_every_stage(self):
        valid = copy.deepcopy(self.config)
        valid["dataset"].update(stage_mode="before_then_after", split_strategy="range_overlap", train_side="before",
                                after_test_fraction=.2, after_grouping="range_bins", after_range_edges_km=[1., 2., 3., 4.])
        valid["continual"]["stages"] = 4
        self.config_module.validate_config(valid)
        invalid_values = (None, "1,2,3,4", [1., 2., 4.], [1., 2., 3., 4., 5.],
                          [0., 2., 3., 4.], [-1., 2., 3., 4.], [1., True, 3., 4.],
                          [1., 2., 2., 4.], [1., 3., 2., 4.], [1., 2., 3., float("inf")],
                          [1., 2., float("nan"), 4.])
        for edges in invalid_values:
            invalid = copy.deepcopy(valid)
            invalid["dataset"]["after_range_edges_km"] = edges
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                self.config_module.validate_config(invalid)
        missing = copy.deepcopy(valid)
        del missing["dataset"]["after_range_edges_km"]
        with self.assertRaises(ValueError):
            self.config_module.validate_config(missing)
        wrong_stage_count = copy.deepcopy(valid)
        wrong_stage_count["continual"]["stages"] = 3
        with self.assertRaises(ValueError):
            self.config_module.validate_config(wrong_stage_count)
        misplaced = copy.deepcopy(valid)
        misplaced["dataset"]["after_grouping"] = "time_interleaved"
        with self.assertRaises(ValueError):
            self.config_module.validate_config(misplaced)
        for key, value in (("after_grouping", "time_interleaved"), ("after_range_edges_km", [1., 2., 3., 4.])):
            invalid = copy.deepcopy(self.config)
            invalid["dataset"][key] = value
            with self.subTest(unsupported_field=key), self.assertRaises(ValueError):
                self.config_module.validate_config(invalid)

    def test_range_group_resolution_keeps_formal_bins_and_reduces_only_quick_bins(self):
        tasks = import_module(f"scripts.{PACK}.localization_networks.workflow.tasks")
        submitted = copy.deepcopy(self.config)
        submitted["dataset"].update(stage_mode="before_then_after", split_strategy="range_overlap",
                                    train_side="before", after_test_fraction=.2, after_grouping="range_bins",
                                    after_range_edges_km=[1., 2., 3., 4.])
        submitted["continual"]["stages"] = 4
        original = copy.deepcopy(submitted)
        for suite in ("quick", "smoke", "experiment"):
            with self.subTest(suite=suite):
                resolved = tasks.resolved_config(submitted, suite)
                self.config_module.validate_config(resolved)
                self.assertEqual(submitted, original)
                self.assertEqual(resolved["dataset"]["stage_mode"], "before_then_after")
                self.assertEqual(resolved["dataset"]["after_grouping"], "range_bins")
                self.assertEqual(resolved["dataset"]["after_test_fraction"], .2)
                self.assertEqual(resolved["dataset"]["train_side"], "before")
                if suite == "quick":
                    self.assertEqual(resolved["continual"]["stages"], 2)
                    self.assertEqual(resolved["dataset"]["after_range_edges_km"], [1., 4.])
                else:
                    self.assertEqual(resolved["continual"]["stages"], 4)
                    self.assertEqual(resolved["dataset"], submitted["dataset"])
                if suite in ("quick", "smoke"):
                    for key in ("initial_epochs", "adapt_epochs", "warmup_epochs"):
                        self.assertLessEqual(resolved["training"][key], 1)
                else:
                    self.assertEqual(resolved, submitted)

    def _distance_holdout_config(self):
        config = copy.deepcopy(self.config)
        config["dataset"].update(stage_mode="distance_holdout", split_strategy="range_overlap",
                                 train_side="before", distance_edges_km=[.90, 1.32, 1.74, 2.16], heldout_group=2)
        config["continual"]["stages"] = 3
        return config

    def test_distance_holdout_requires_valid_edges_group_and_exclusive_mode_fields(self):
        valid = self._distance_holdout_config()
        for edges in ([1., 2., 3.], [.90, 1.32, 1.74, 2.16], [1., 2., 3., 4., 5.]):
            for heldout in range(1, len(edges)):
                accepted = copy.deepcopy(valid)
                accepted["dataset"].update(distance_edges_km=edges, heldout_group=heldout)
                accepted["continual"]["stages"] = len(edges) - 1
                with self.subTest(edges=edges, heldout=heldout):
                    self.config_module.validate_config(accepted)
        for edges in (None, "1,2,3,4", [1., 2.], [0., 2., 3., 4.], [-1., 2., 3., 4.],
                      [1., True, 3., 4.], [1., 2., 2., 4.], [1., 3., 2., 4.],
                      [1., 2., 3., float("inf")], [1., 2., float("nan"), 4.]):
            invalid = copy.deepcopy(valid)
            invalid["dataset"]["distance_edges_km"] = edges
            with self.subTest(invalid_edges=edges), self.assertRaises(ValueError):
                self.config_module.validate_config(invalid)
        for heldout in (0, 4, -1, True, 1.0, "1", None):
            invalid = copy.deepcopy(valid)
            invalid["dataset"]["heldout_group"] = heldout
            with self.subTest(invalid_heldout=heldout), self.assertRaises(ValueError):
                self.config_module.validate_config(invalid)
        for missing in ("distance_edges_km", "heldout_group"):
            invalid = copy.deepcopy(valid)
            del invalid["dataset"][missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.config_module.validate_config(invalid)
        for key, value in (("after_test_fraction", .2), ("after_grouping", "range_bins"),
                           ("after_range_edges_km", [.90, 1.32, 1.74, 2.16]),
                           ("train_side", "after"), ("split_strategy", "full_interleaved")):
            invalid = copy.deepcopy(valid)
            invalid["dataset"][key] = value
            with self.subTest(invalid_field=key), self.assertRaises(ValueError):
                self.config_module.validate_config(invalid)
        invalid = copy.deepcopy(valid)
        invalid["continual"]["stages"] = 4
        with self.assertRaises(ValueError):
            self.config_module.validate_config(invalid)
        for key in ("heldout_group", "distance_edges_km"):
            invalid = copy.deepcopy(self.config)
            invalid["dataset"][key] = valid["dataset"][key]
            with self.subTest(misplaced=key), self.assertRaises(ValueError):
                self.config_module.validate_config(invalid)

    def test_distance_holdout_resolution_preserves_real_protocol_and_marks_quick_synthetic(self):
        tasks = import_module(f"scripts.{PACK}.localization_networks.workflow.tasks")
        submitted = self._distance_holdout_config()
        original = copy.deepcopy(submitted)
        for suite in ("quick", "smoke", "experiment"):
            with self.subTest(suite=suite):
                resolved = tasks.resolved_config(submitted, suite)
                self.config_module.validate_config(resolved)
                self.assertEqual(submitted, original)
                if suite == "quick":
                    self.assertEqual(resolved["dataset"]["stage_mode"], "interleaved")
                    self.assertEqual(resolved["continual"]["stages"], 2)
                    self.assertNotIn("distance_edges_km", resolved["dataset"])
                    self.assertNotIn("heldout_group", resolved["dataset"])
                else:
                    self.assertEqual(resolved["dataset"], submitted["dataset"])
                    self.assertEqual(resolved["continual"]["stages"], 3)
                for key in ("after_test_fraction", "after_grouping", "after_range_edges_km"):
                    self.assertNotIn(key, resolved["dataset"])
                if suite in ("quick", "smoke"):
                    for key in ("initial_epochs", "adapt_epochs", "warmup_epochs"):
                        self.assertLessEqual(resolved["training"][key], 1)
                else:
                    self.assertEqual(resolved, submitted)

    def test_near_middle_far_configs_differ_only_in_complete_heldout_distance_group(self):
        reference = None
        for heldout, label in enumerate(("near", "middle", "far"), 1):
            with self.subTest(label=label):
                config = self.remote.load_config(self.remote.DEFAULT_CONFIG.with_name(f"haiqin_holdout_{label}.toml"))
                self.assertEqual(config["dataset"]["stage_mode"], "distance_holdout")
                self.assertEqual(config["dataset"]["distance_edges_km"], [.90, 1.32, 1.74, 2.16])
                self.assertEqual(config["dataset"]["heldout_group"], heldout)
                self.assertEqual(config["continual"]["stages"], 3)
                comparable = copy.deepcopy(config)
                comparable["dataset"].pop("heldout_group")
                if reference is None:
                    reference = comparable
                else:
                    self.assertEqual(comparable, reference)

    def test_invalid_custom_config_fails_before_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config with spaces.toml"
            path.write_text(self.remote.DEFAULT_CONFIG.read_text() + "\nobsolete = true\n")
            with patch.object(self.remote, "dispatch") as dispatch, \
                    patch.object(self.remote, "remote_python") as ssh, \
                    patch("sys.stderr", io.StringIO()):
                self.assertEqual(main([PACK, "--config", str(path), "--dry-run"]), 1)
            dispatch.assert_not_called()
            ssh.assert_not_called()

    def test_help_and_dry_run_import_only_standard_library(self):
        program = '''import importlib.abc,sys
class RejectNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','numpy','scipy','h5py','matplotlib'}: raise AssertionError(fullname)
sys.meta_path.insert(0, RejectNumerical())
from scripts.run import main
try:
 main(['MoE_learn','--help'])
except SystemExit as exc:
 assert exc.code == 0
assert main(['MoE_learn','--dry-run']) == 0
'''
        result = subprocess.run([sys.executable, "-B", "-c", program], cwd=PROJECT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(NETWORK, result.stdout)
        self.assertIn("demo", result.stdout)

    def test_quick_preflight_and_initialization_allow_absent_original_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repository, workspace, uv = root / "repository", root / "workspace", root / "uv"
            repository.mkdir()
            uv.write_text("fixture; not executed")
            config = copy.deepcopy(self.config)
            config["remote"].update(repository=str(repository), workspace=str(workspace), uv=str(uv))
            for require_raw, expected in ((False, "PASS"), (True, "FAIL")):
                with patch("sys.stdin", io.StringIO(json.dumps({**config, "require_raw": require_raw}))), \
                        patch("sys.stdout", io.StringIO()) as output, \
                        patch("subprocess.check_output", return_value="uv fixture"):
                    exec(self.remote.CHECK_CODE, {"__name__": "__main__"})
                result = json.loads(output.getvalue())
                self.assertEqual(result["status"], expected)
                self.assertFalse(result["origindata_exists"])
            for suite in ("quick", "smoke"):
                run_id = make_run_id(PACK, config)
                payload = {"workspace": str(workspace), "repository": str(repository),
                           "run_id": run_id, "dispatch_token": "fixture", "report": {"status": "STARTING"},
                           "manifest.json": {}, "request.json": {"suite": suite}, "resolved-config.json": config}
                with patch("sys.stdin", io.StringIO(json.dumps(payload))), patch("sys.stdout", io.StringIO()):
                    if suite == "smoke":
                        with self.assertRaisesRegex(ValueError, "original data"):
                            exec(self.remote.INIT_CODE, {"__name__": "__main__"})
                    else:
                        exec(self.remote.INIT_CODE, {"__name__": "__main__"})
                        self.assertTrue((workspace / "runs" / run_id / "request.json").is_file())
                        self.assertFalse((workspace / "runs" / run_id / "project" / "Origindata").exists())

    def test_quick_worker_never_invokes_raw_identity_phase(self):
        worker_module = import_module(f"scripts.{PACK}.localization_networks.workflow.worker")
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            worker = worker_module.Worker(workspace / "runs" / "fixture")
            worker.request = {"suite": "quick"}
            worker.config = {"remote": {"workspace": str(workspace)}}
            worker.manifest = {}
            with patch.object(worker, "setup", return_value={}), \
                    patch.object(worker, "environment", return_value={}), \
                    patch.object(worker, "demo", return_value={}), \
                    patch.object(worker, "data_identity", side_effect=AssertionError("quick read original data")) as raw, \
                    patch.object(worker_module, "verify_snapshot"), patch("sys.stdout", io.StringIO()):
                self.assertEqual(worker.execute(), 0)
            raw.assert_not_called()
            self.assertEqual(worker.report["status"], "PASS")
            self.assertNotIn("original_data_identity", {row["name"] for row in worker.report["checks"]})

    def test_worker_export_and_fetch_verify_checkpoint_bytes_without_network(self):
        worker_module = import_module(f"scripts.{PACK}.localization_networks.workflow.worker")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = make_run_id(PACK, self.config)
            worker = worker_module.Worker(root / "remote" / run_id)
            demo = worker.run / "reports" / "demo"
            demo.mkdir(parents=True)
            (demo / "checkpoint.pt").write_bytes(b"fixture checkpoint bytes")
            (demo / "demo-report.json").write_text(json.dumps({"pack": PACK, "status": "PASS"}))
            (demo / "stages.csv").write_text("stage,action\n1,INITIAL_TRAIN\n")
            (demo / "private.h5").write_bytes(b"not exported")
            (demo / "summary.pdf").write_bytes(b"fixture figure; no plotting")
            (worker.run / "logs").mkdir()
            (worker.run / "logs" / "worker.log").write_text("fixture log")
            report = {"schema_version": 1, "pack": PACK, "run_id": run_id,
                      "status": "PASS", "suite": "experiment", "checks": []}
            worker.export(report)
            local, figures = root / "local-data", root / "local-figures"

            def copy_transfer(argv, **kwargs):
                self.assertEqual(argv[0], "rsync")
                self.assertIn("--checksum", argv)
                remote_source, destination = argv[-2:]
                if "/export/data/" in remote_source:
                    source = worker.run / "export/data"
                elif "/export/figure/" in remote_source:
                    source = worker.run / "export/figure"
                else:
                    self.assertTrue(remote_source.endswith("/logs/"))
                    source = worker.run / "logs"
                shutil.copytree(source, destination, dirs_exist_ok=True)
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(self.remote, "status", return_value=report), \
                    patch.object(self.remote, "local_paths", return_value=(local, figures)), \
                    patch.object(self.remote.subprocess, "run", side_effect=copy_transfer), \
                    patch("sys.stdout", io.StringIO()):
                self.assertEqual(self.remote.fetch(self.config, run_id), report)
            checkpoint = local / "reports/demo/checkpoint.pt"
            self.assertEqual(checkpoint.read_bytes(), b"fixture checkpoint bytes")
            self.assertFalse((local / "reports/demo/private.h5").exists())
            self.assertTrue((figures / "reports/demo/summary.pdf").is_file())
            verification = json.loads((local / "delivery-verification.json").read_text())
            self.assertEqual(verification["status"], "PASS")
            # Keep the same length: verification must compare content, not only size.
            checkpoint.write_bytes(b"Fixture checkpoint bytes")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.remote.verify_artifact_delivery(local, figures, run_id)

    def test_artifact_manifest_rejects_cross_pack_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local, figures = root / "data", root / "figures"
            local.mkdir()
            figures.mkdir()
            run_id = make_run_id(PACK, self.config)
            path = local / "artifact-manifest.json"
            for pack, file_path in (("MoE_data_scale", "report.json"), (PACK, "../outside.json")):
                manifest = {"schema_version": 1, "pack": pack, "run_id": run_id,
                            "files": [{"group": "data", "path": file_path, "size": 0, "sha256": "a" * 64}]}
                path.write_text(json.dumps(manifest))
                with self.subTest(pack=pack, path=file_path), self.assertRaises(ValueError):
                    self.remote.verify_artifact_delivery(local, figures, run_id)


if __name__ == "__main__":
    unittest.main()
