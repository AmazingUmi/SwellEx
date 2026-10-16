"""No numerical work: budget orchestration uses immutable fixtures and mocked SSH."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.data_scale_pack.localization_networks.workflow import budget_study as budget, remote, study
from scripts.data_scale_pack.localization_networks.workflow.tasks import model_plan
from scripts.data_scale_pack.localization_networks.workflow.worker import expected_controlled_budget


class BudgetStudyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swellex-budget-study-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs = self.root / "data"
        self.runs.mkdir()
        self.directory = self.runs / "budget-study"
        self.plan_path = self.root / "plan.json"
        self.project = self.root / "project"
        self.artifact_path = "scripts/data_scale_pack/localization_networks/workflow/calibration_fixture.json"
        artifact_file = self.project / self.artifact_path
        artifact_file.parent.mkdir(parents=True)
        self.artifact = {"schema_version": 1, "development_seeds": [2026, 2027, 2028],
                         "policy": {"reference_samples": 600, "exponent": 0.75}}
        self.raw = (json.dumps(self.artifact, indent=2) + "\n").encode()
        artifact_file.write_bytes(self.raw)
        calibration = {"path": self.artifact_path, "sha256": hashlib.sha256(self.raw).hexdigest()}
        config = remote.load_config(remote.DEFAULT_CONFIG)
        config["training"].update(batch_size=8, epochs=2)
        self.plan = {"schema_version": 1, "base_config": config, "fractions": [.5, 1.],
                     "seeds": [2029], "method": "SCM", "epoch_budget": 2, "update_budget": 16,
                     "validation_interval_updates": 8, "calibration": calibration}
        self.write_plan()
        self.files = [{"path": self.artifact_path, "size": len(self.raw), "sha256": calibration["sha256"]}]
        self.source_hash = study.digest(self.files)
        self.full_count = 64
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(remote, "PROJECT", self.project))
        stack.enter_context(mock.patch.object(remote, "LOCAL_RUNS", self.runs))
        stack.enter_context(mock.patch.object(remote, "LOCAL_FIGURES", self.root / "figure"))
        self.inventory = stack.enter_context(mock.patch.object(study, "source_inventory", return_value=(self.files, self.source_hash)))
        self.dispatch = stack.enter_context(mock.patch.object(remote, "dispatch", side_effect=self.fake_dispatch))
        self.status = stack.enter_context(mock.patch.object(remote, "status", side_effect=self.read_report))
        self.fetch = stack.enter_context(mock.patch.object(remote, "fetch", side_effect=self.read_report))
        self.follow = stack.enter_context(mock.patch.object(remote, "follow_run", side_effect=self.fake_follow))
        stack.enter_context(mock.patch.object(remote, "verify_visualization_delivery", return_value={"status": "PASS"}))
        self.output = stack.enter_context(mock.patch("sys.stdout", io.StringIO()))

    def write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan))

    def ledger(self):
        return study.read_json(self.directory / "study.json")

    def read_report(self, config, run_id):
        return study.read_json(self.directory / "children" / run_id / "report.json")

    def fake_dispatch(self, config, suite, no_cache, *, selection, run_id, expected_source_sha256, max_connection_failures):
        directory = self.directory / "children" / run_id
        directory.mkdir(parents=True)
        common = {"pack": budget.PACK, "run_id": run_id, "suite": suite, "selection": selection,
                  "source_sha256": expected_source_sha256, "config_sha256": study.digest(config, compact=False)}
        remote.dump(directory / "request.json", {"schema_version": 1, "pack": budget.PACK, "run_id": run_id,
                                                 "suite": suite, "selection": selection, "config": config, "use_cache": not no_cache})
        remote.dump(directory / "manifest.json", {**common, "files": self.files})
        count = int(self.full_count * config["training"]["train_fraction"])
        observed = expected_controlled_budget(config["training"], count)
        jobs = model_plan(suite, selection)
        rows = [{"method": job["method"], "model": job["model"], "scaling": {"train_count": count},
                 "budget_plan": observed, "optimizer_steps": observed["total_optimizer_updates"],
                 "validation_checks": observed["validation_checks"], "unique_train_sources_seen": count} for job in jobs]
        remote.dump(directory / "report.json", {**common, "status": "PASS", "model_plan": jobs,
                                                "calibration": config["calibration"], "metrics": rows})
        return 0

    def fake_follow(self, config, run_id, *, max_connection_failures):
        report = self.read_report(config, run_id)
        report["status"] = "PASS"
        remote.dump(self.directory / "children" / run_id / "report.json", report)
        return 0

    def run_study(self, **kwargs):
        return budget.run_budget_study(self.plan_path, study_dir=self.directory, **kwargs)

    def test_formal_matrix_has_105_unique_training_observations_and_120_analysis_cells(self):
        self.plan.update(fractions=[.125, .25, .5, 1.], seeds=[2029, 2030, 2031],
                         epoch_budget=100, update_budget=4200, validation_interval_updates=42)
        ledger = budget.make_ledger(budget.validate_plan(self.plan), self.directory, self.artifact)
        self.assertEqual(len(ledger["children"]), 21)
        self.assertEqual(len(ledger["analysis_cells"]), 24)
        self.assertEqual(len({c["run_id"] for c in ledger["children"]}), 21)
        for seed in self.plan["seeds"]:
            cells = [c for c in ledger["analysis_cells"] if c["seed"] == seed and c["train_fraction"] == 1]
            self.assertEqual(len({c["run_id"] for c in cells}), 1)
            self.assertTrue(all(c["shared"] and c["shared_reason"] == budget.SHARED_REASON for c in cells))

    def test_dispatches_once_and_freezes_raw_artifact_and_actual_budget(self):
        self.assertEqual(self.run_study(), 0)
        ledger = self.ledger()
        self.assertEqual(ledger["status"], "PASS")
        self.assertEqual(self.dispatch.call_count, 3)
        self.assertEqual((self.directory / "calibration.json").read_bytes(), self.raw)
        self.assertEqual(ledger["calibration_artifact"], self.artifact)
        self.assertEqual([c["shared_full_verified"] for c in ledger["children"]], [False, False, True])
        for call in self.dispatch.call_args_list:
            config, suite, no_cache = call.args
            self.assertEqual(suite, "budget")
            self.assertFalse(no_cache)
            self.assertTrue(config["training"]["controlled_batches"])
            self.assertEqual(config["scaling"], budget.OLD_POLICY)
            self.assertEqual(config["calibration"], self.plan["calibration"])
            self.assertEqual(call.kwargs["expected_source_sha256"], self.source_hash)
            self.assertEqual(call.kwargs["max_connection_failures"], 3)

    def test_dry_run_has_no_output_or_remote_effects(self):
        self.assertEqual(self.run_study(dry_run=True), 0)
        self.assertFalse(self.directory.exists())
        self.dispatch.assert_not_called()
        self.inventory.assert_not_called()
        result = json.loads(self.output.getvalue())
        self.assertEqual(result["unique_model_trainings"], 15)
        self.assertEqual(result["analysis_observations"], 20)
        self.assertEqual(result["status"], "NOT_RUN")

    def test_reentry_verifies_all_pass_without_retraining(self):
        self.run_study()
        self.dispatch.reset_mock()
        self.assertEqual(self.run_study(), 0)
        self.dispatch.assert_not_called()
        self.assertEqual(self.fetch.call_count, 3)
        self.follow.assert_not_called()

    def test_interrupted_existing_child_is_followed_without_redispatch(self):
        def interrupted(*args, **kwargs):
            self.fake_dispatch(*args, **kwargs)
            report = self.read_report(args[0], kwargs["run_id"])
            report["status"] = "RUNNING"
            remote.dump(self.directory / "children" / kwargs["run_id"] / "report.json", report)
            return 130
        self.dispatch.side_effect = interrupted
        self.assertEqual(self.run_study(), 130)
        first = self.ledger()["children"][0]["run_id"]
        self.dispatch.reset_mock(side_effect=True)
        self.dispatch.side_effect = self.fake_dispatch
        self.assertEqual(self.run_study(), 0)
        self.assertEqual(self.dispatch.call_count, 2)
        self.assertEqual(self.follow.call_count, 1)
        self.assertEqual(self.follow.call_args.args[1], first)

    def test_failed_child_is_never_rerun(self):
        def failed(*args, **kwargs):
            self.fake_dispatch(*args, **kwargs)
            report = self.read_report(args[0], kwargs["run_id"])
            report["status"] = "FAIL"
            remote.dump(self.directory / "children" / kwargs["run_id"] / "report.json", report)
            return 1
        self.dispatch.side_effect = failed
        self.assertEqual(self.run_study(), 1)
        self.dispatch.reset_mock()
        self.assertEqual(self.run_study(), 1)
        self.dispatch.assert_not_called()

    def test_missing_launch_marker_does_not_relaunch_or_follow(self):
        def interrupted(*args, **kwargs):
            self.fake_dispatch(*args, **kwargs)
            report = self.read_report(args[0], kwargs["run_id"])
            report["status"] = "STARTING"
            remote.dump(self.directory / "children" / kwargs["run_id"] / "report.json", report)
            return 130
        self.dispatch.side_effect = interrupted
        self.assertEqual(self.run_study(), 130)
        self.dispatch.reset_mock()
        with mock.patch.object(remote, "remote_python", return_value={"launched": False}), self.assertRaisesRegex(RuntimeError, "launch marker"):
            self.run_study()
        self.dispatch.assert_not_called()
        self.follow.assert_not_called()

    def test_actual_full_count_must_prove_shared_schedule(self):
        self.full_count = 72
        with self.assertRaisesRegex(ValueError, "cannot be shared"):
            self.run_study()
        self.assertEqual(self.ledger()["children"][-1]["status"], "UNKNOWN")
        self.assertNotIn("shared_full_verified", self.ledger()["children"][-1])

    def test_source_drift_stops_before_dispatch(self):
        self.inventory.side_effect = [(self.files, self.source_hash), (self.files, "a" * 64)]
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.run_study()
        self.dispatch.assert_not_called()

    def test_calibration_hash_development_seeds_and_frozen_copy_are_checked(self):
        self.plan["seeds"] = [2026]
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.run_study()
        self.plan["seeds"] = [2029]
        self.plan["calibration"]["sha256"] = "a" * 64
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "byte SHA256"):
            self.run_study()
        self.plan["calibration"]["sha256"] = hashlib.sha256(self.raw).hexdigest()
        self.write_plan()
        self.run_study()
        (self.directory / "calibration.json").write_bytes(self.raw + b"\n")
        with self.assertRaisesRegex(ValueError, "Frozen calibration"):
            self.run_study()

    def test_plan_and_analysis_alias_drift_are_rejected(self):
        self.run_study()
        self.plan["seeds"] = [2030]
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            self.run_study()
        self.plan["seeds"] = [2029]
        self.write_plan()
        ledger = self.ledger()
        ledger["analysis_cells"][-1]["run_id"] = ledger["children"][0]["run_id"]
        study.save(self.directory, ledger)
        with self.assertRaisesRegex(ValueError, "shared observation"):
            self.run_study()

    def test_pass_source_and_budget_tampering_are_rejected(self):
        self.run_study()
        child = self.ledger()["children"][0]
        report_path = self.directory / "children" / child["run_id"] / "report.json"
        report = study.read_json(report_path)
        report["metrics"][0]["optimizer_steps"] += 1
        remote.dump(report_path, report)
        with self.assertRaisesRegex(ValueError, "exposure"):
            self.run_study()
        self.dispatch.reset_mock()
        report["metrics"][0]["optimizer_steps"] -= 1
        report["source_sha256"] = "f" * 64
        remote.dump(report_path, report)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_study()
        self.dispatch.assert_not_called()

    def test_unknown_plan_and_invalid_budget_combinations_are_rejected(self):
        invalid_values = [("method", None), ("method", "ELM"), ("seeds", [True]),
                          ("fractions", [.5]), ("fractions", [True, .5]), ("fractions", [.5, .5, 1]),
                          ("epoch_budget", False), ("update_budget", 17), ("validation_interval_updates", 4)]
        for key, value in invalid_values:
            plan = copy.deepcopy(self.plan)
            plan[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                budget.validate_plan(plan)
        plan = copy.deepcopy(self.plan)
        plan["profiles"] = ["small"]
        with self.assertRaises(ValueError):
            budget.validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
