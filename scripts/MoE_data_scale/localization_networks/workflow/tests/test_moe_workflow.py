"""Focused stdlib contracts for isolated MoE architecture validation."""
from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.MoE_data_scale.localization_networks.workflow import remote
from scripts.MoE_data_scale.localization_networks.workflow.config import load_config, validate_config
from scripts.MoE_data_scale.localization_networks.workflow.tasks import expected_phases, model_plan
from scripts.MoE_data_scale.localization_networks.workflow.worker import Worker
from scripts.run_identity import make_run_id


class MoEWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(remote.DEFAULT_CONFIG)

    def test_all_presets_use_expert_policy_and_balance_loss(self):
        for path in remote.DEFAULT_CONFIG.parent.glob("*.toml"):
            with self.subTest(path=path):
                config = load_config(path)
                self.assertEqual(config["scaling"], {"reference_samples": 600, "reference_experts": 2,
                                                     "exponent": 0.5, "max_experts": 8})
                self.assertEqual(config["moe"], {"balance_loss_weight": 0.01})

    def test_unknown_and_obsolete_fields_are_rejected(self):
        mutations = [lambda c: c.update(calibration={"path": "old.json", "sha256": "a"*64}),
                     lambda c: c["training"].update(controlled_batches=True),
                     lambda c: c["training"].update(optimizer_updates=10),
                     lambda c: c["scaling"].update(depth=4),
                     lambda c: c["moe"].update(top_k=2)]
        for mutate in mutations:
            config = copy.deepcopy(self.config)
            mutate(config)
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_expert_counts_and_balance_weight_are_strict(self):
        for table, key, value in (("scaling", "reference_samples", True),
                                  ("scaling", "reference_experts", 3),
                                  ("scaling", "max_experts", 16),
                                  ("scaling", "exponent", float("nan")),
                                  ("moe", "balance_loss_weight", -0.01),
                                  ("moe", "balance_loss_weight", True)):
            config = copy.deepcopy(self.config)
            config[table][key] = value
            with self.subTest(table=table, key=key, value=value), self.assertRaises(ValueError):
                validate_config(config)
        self.config["scaling"].update(reference_experts=8, max_experts=4)
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_quick_has_only_synthetic_runtime_phases(self):
        self.assertEqual(expected_phases("quick"), {"snapshot_integrity", "environment", "python_quick",
                                                   "source_unchanged_after_validation"})
        self.assertEqual(model_plan("quick"), [])
        with self.assertRaises(ValueError):
            model_plan("budget", {"method": "SCM", "network": None})

    def test_cli_receives_every_frozen_expert_and_loss_option(self):
        worker = Worker(Path(tempfile.gettempdir()) / "moe-contract-run")
        worker.config = self.config
        options = worker.scaling_arguments()
        expected = {"--train-fraction": "1.0", "--scale-reference-samples": "600",
                    "--scale-reference-experts": "2", "--scale-exponent": "0.5",
                    "--scale-max-experts": "8", "--balance-loss-weight": "0.01"}
        self.assertEqual(dict(zip(options[::2], options[1::2])), expected)

    def test_delivery_paths_are_top_level_and_ignore_other_study_ledgers(self):
        run_id = make_run_id(remote.PACK, self.config)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "other").mkdir()
            (root / "other/study.json").write_text("bad unrelated ledger")
            with patch.object(remote, "LOCAL_RUNS", root), patch.object(remote, "LOCAL_FIGURES", root / "figures"):
                self.assertEqual(remote.local_paths(run_id), (root / run_id, root / "figures" / run_id))

    def test_snapshot_includes_untracked_new_pack_and_excludes_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            paths = [Path("scripts/MoE_data_scale/localization_networks/workflow/worker.py"),
                     Path("scripts/MoE_data_scale/localization_networks/common/new_model.py"),
                     Path("scripts/MoE_data_scale/localization_networks/__pycache__/cache.pyc"),
                     Path("outputs/artifact.json")]
            for path in paths:
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text("test source")
            sources = remote.source_paths(root)
            self.assertEqual(sources, sorted(paths[:2]))

    def test_quick_preflight_and_initialization_need_no_original_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            repository = root / "repository"
            repository.mkdir()
            uv = root / "uv"
            uv.write_text("#!/bin/sh\necho uv-test\n")
            uv.chmod(0o755)
            payload = {"require_raw": False, "remote": {"repository": str(repository), "uv": str(uv)}}
            result = subprocess.run([sys.executable, "-B", "-c", remote.CHECK_CODE],
                                    input=json.dumps(payload), capture_output=True, text=True, check=True)
            preflight = json.loads(result.stdout)
            self.assertEqual(preflight["status"], "PASS")
            self.assertFalse(preflight["origindata_exists"])
            run_id = make_run_id(remote.PACK, self.config)
            payload = {"workspace": str(root / "workspace"), "repository": str(repository),
                       "run_id": run_id, "dispatch_token": "test", "report": {},
                       "manifest.json": {}, "request.json": {"suite": "quick"}, "generation_config.json": {}}
            result = subprocess.run([sys.executable, "-B", "-c", remote.INIT_CODE],
                                    input=json.dumps(payload), capture_output=True, text=True, check=True)
            self.assertTrue(Path(json.loads(result.stdout)["run_dir"]).is_dir())

    def test_quick_worker_never_enters_data_training_or_plot_phases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            worker = Worker(root / "run")
            worker.run.mkdir()
            worker.config = copy.deepcopy(self.config)
            worker.config["remote"]["workspace"] = str(root)
            worker.request = {"suite": "quick"}
            worker.manifest = {}
            worker.python = sys.executable
            with patch.object(worker, "setup", return_value={}), \
                    patch.object(worker, "environment", return_value={}), \
                    patch.object(worker, "json_command", return_value={"status": "PASS"}) as quick, \
                    patch.object(worker, "data_identity", side_effect=AssertionError("raw data accessed")), \
                    patch.object(worker, "datasets", side_effect=AssertionError("generation entered")), \
                    patch.object(worker, "train_predict", side_effect=AssertionError("training entered")), \
                    patch.object(worker, "visualize", side_effect=AssertionError("plotting entered")), \
                    patch("scripts.MoE_data_scale.localization_networks.workflow.worker.verify_snapshot"), \
                    patch("sys.stdout", io.StringIO()):
                self.assertEqual(worker.execute(), 0)
            self.assertEqual(quick.call_count, 1)
            self.assertEqual(quick.call_args.args[0], "python_quick")
            self.assertEqual({row["name"] for row in worker.report["checks"]}, expected_phases("quick"))
            self.assertEqual(worker.report["status"], "PASS")
            self.assertEqual(worker.report["not_run"], [])


if __name__ == "__main__":
    unittest.main()
