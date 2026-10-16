"""Study orchestration tests use only standard-library fixtures and mocked SSH."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import fcntl
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.data_scale_pack.localization_networks.workflow import remote, study, study_remote
from scripts.run_identity import make_run_id


class StudyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swellex-study-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs = self.root / "data"
        self.runs.mkdir()
        self.directory = self.runs / "controlled-study"
        self.plan_path = self.root / "plan.json"
        self.config = remote.load_config(remote.DEFAULT_CONFIG)
        self.plan = {"schema_version": 1, "base_config": self.config,
                     "fractions": [0.5, 1.0], "seeds": [2026, 2027], "method": "SCM"}
        self.write_plan()
        self.files = [{"path": "scripts/fixture.py", "size": 10, "sha256": "a" * 64}]
        self.source_sha256 = study.digest(self.files)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(remote, "LOCAL_RUNS", self.runs))
        stack.enter_context(mock.patch.object(remote, "LOCAL_FIGURES", self.root / "figure"))
        self.inventory = stack.enter_context(mock.patch.object(study, "source_inventory", return_value=(self.files, self.source_sha256)))
        self.dispatch = stack.enter_context(mock.patch.object(remote, "dispatch", side_effect=self.fake_dispatch))
        self.status = stack.enter_context(mock.patch.object(remote, "status", side_effect=self.read_report))
        self.fetch = stack.enter_context(mock.patch.object(remote, "fetch", side_effect=self.read_report))
        self.follow = stack.enter_context(mock.patch.object(remote, "follow_run", side_effect=self.fake_follow))
        stack.enter_context(mock.patch.object(remote, "verify_visualization_delivery", return_value={"status": "PASS"}))
        stack.enter_context(mock.patch("sys.stdout", io.StringIO()))

    def write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan))

    def ledger(self):
        return study.read_json(self.directory / "study.json")

    def read_report(self, config, run_id):
        return study.read_json(self.directory / "children" / run_id / "report.json")

    def fake_dispatch(self, config, suite, no_cache, *, selection, run_id, expected_source_sha256, max_connection_failures):
        directory = self.directory / "children" / run_id
        directory.mkdir(parents=True)
        request = {"schema_version": 1, "pack": study.PACK, "run_id": run_id,
                   "suite": suite, "config": config, "use_cache": not no_cache}
        common = {"pack": study.PACK, "run_id": run_id, "suite": suite,
                  "source_sha256": expected_source_sha256,
                  "config_sha256": study.digest(config, compact=False)}
        if selection is not None:
            request["selection"] = selection
            common["selection"] = selection
        remote.dump(directory / "request.json", request)
        remote.dump(directory / "manifest.json", {**common, "files": self.files})
        remote.dump(directory / "report.json", {**common, "status": "PASS"})
        return 0

    def fake_follow(self, config, run_id, *, max_connection_failures):
        report = self.read_report(config, run_id)
        report["status"] = "PASS"
        remote.dump(self.directory / "children" / run_id / "report.json", report)
        return 0

    def run_study(self):
        return study.run_study(self.plan_path, study_dir=self.directory)

    def test_explicit_new_directory_freezes_complete_matrix_and_identity(self):
        self.assertEqual(self.run_study(), 0)
        ledger = self.ledger()
        self.assertEqual(ledger["status"], "PASS")
        self.assertEqual(ledger["source_sha256"], self.source_sha256)
        self.assertEqual(ledger["plan_sha256"], study.digest(self.plan))
        self.assertEqual(len(ledger["children"]), 4)
        self.assertEqual([(c["seed"], c["train_fraction"]) for c in ledger["children"]],
                         [(2026, .5), (2026, 1.), (2027, .5), (2027, 1.)])
        self.assertEqual(self.dispatch.call_count, 4)
        for call, child in zip(self.dispatch.call_args_list, ledger["children"], strict=True):
            config, suite, no_cache = call.args
            self.assertEqual(suite, "selected")
            self.assertFalse(no_cache)
            self.assertEqual(config["validation"]["seed"], child["seed"])
            self.assertEqual(config["training"]["train_fraction"], child["train_fraction"])
            self.assertEqual(call.kwargs["selection"], {"method": "SCM", "network": None})
            self.assertEqual(call.kwargs["run_id"], child["run_id"])
            self.assertEqual(call.kwargs["expected_source_sha256"], self.source_sha256)
            self.assertEqual(call.kwargs["max_connection_failures"], 3)

    def test_null_method_uses_all_eighteen_model_experiment(self):
        self.plan.update(method=None, fractions=[1.0], seeds=[2026])
        self.write_plan()
        self.assertEqual(self.run_study(), 0)
        self.assertEqual(self.dispatch.call_args.args[1], "experiment")
        self.assertIsNone(self.dispatch.call_args.kwargs["selection"])

    def test_reentry_verifies_pass_children_without_dispatching_again(self):
        self.assertEqual(self.run_study(), 0)
        self.dispatch.reset_mock()
        self.assertEqual(self.run_study(), 0)
        self.dispatch.assert_not_called()
        self.assertEqual(self.fetch.call_count, 4)
        self.follow.assert_not_called()

    def test_interrupted_child_is_followed_then_remaining_children_launch_once(self):
        def interrupted(*args, **kwargs):
            self.fake_dispatch(*args, **kwargs)
            report = self.read_report(args[0], kwargs["run_id"])
            report["status"] = "RUNNING"
            remote.dump(self.directory / "children" / kwargs["run_id"] / "report.json", report)
            return 130
        self.dispatch.side_effect = interrupted
        self.assertEqual(self.run_study(), 130)
        first = self.ledger()["children"][0]["run_id"]
        self.assertEqual(self.dispatch.call_count, 1)
        self.dispatch.reset_mock(side_effect=True)
        self.dispatch.side_effect = self.fake_dispatch
        self.assertEqual(self.run_study(), 0)
        self.assertEqual(self.dispatch.call_count, 3)
        self.assertEqual(self.follow.call_count, 1)
        self.assertEqual(self.follow.call_args.args[1], first)
        self.assertTrue(all(row["status"] == "PASS" for row in self.ledger()["children"]))

    def test_ambiguous_launch_is_never_redispatched(self):
        def ambiguous(*args, **kwargs):
            self.fake_dispatch(*args, **kwargs)
            raise RuntimeError("Transport ended after remote launch")
        self.dispatch.side_effect = ambiguous
        with self.assertRaisesRegex(RuntimeError, "Transport"):
            self.run_study()
        self.assertEqual(self.ledger()["children"][0]["status"], "UNKNOWN")
        self.dispatch.reset_mock(side_effect=True)
        self.dispatch.side_effect = self.fake_dispatch
        self.assertEqual(self.run_study(), 0)
        self.assertEqual(self.dispatch.call_count, 3)
        self.assertEqual(self.fetch.call_count, 1)

    def test_unlaunched_starting_child_does_not_wait_or_redispatch(self):
        def abandoned(*args, **kwargs):
            self.fake_dispatch(*args, **kwargs)
            report = self.read_report(args[0], kwargs["run_id"])
            report["status"] = "STARTING"
            remote.dump(self.directory / "children" / kwargs["run_id"] / "report.json", report)
            return 130
        self.dispatch.side_effect = abandoned
        self.assertEqual(self.run_study(), 130)
        self.dispatch.reset_mock()
        with mock.patch.object(remote, "remote_python", return_value={"launched": False}), \
                self.assertRaisesRegex(RuntimeError, "no launch marker"):
            self.run_study()
        self.dispatch.assert_not_called()
        self.follow.assert_not_called()

    def test_failed_child_stops_matrix_and_cannot_be_silently_rerun(self):
        def failed(*args, **kwargs):
            self.fake_dispatch(*args, **kwargs)
            report = self.read_report(args[0], kwargs["run_id"])
            report["status"] = "FAIL"
            remote.dump(self.directory / "children" / kwargs["run_id"] / "report.json", report)
            return 1
        self.dispatch.side_effect = failed
        self.assertEqual(self.run_study(), 1)
        self.assertEqual(self.ledger()["status"], "FAIL")
        self.dispatch.reset_mock()
        self.assertEqual(self.run_study(), 1)
        self.dispatch.assert_not_called()

    def test_source_drift_is_rejected_before_child_launch(self):
        self.inventory.side_effect = [(self.files, self.source_sha256), (self.files, "b" * 64)]
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.run_study()
        self.dispatch.assert_not_called()
        self.assertEqual(self.ledger()["children"][0]["status"], "PLANNED")

    def test_changed_plan_and_pass_source_are_rejected(self):
        self.run_study()
        self.plan["fractions"] = [0.25, 1.0]
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            self.run_study()
        self.plan["fractions"] = [0.5, 1.0]
        self.write_plan()
        run_id = self.ledger()["children"][0]["run_id"]
        manifest_path = self.directory / "children" / run_id / "manifest.json"
        manifest = study.read_json(manifest_path)
        manifest["source_sha256"] = "b" * 64
        remote.dump(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "source/configuration identity"):
            self.run_study()

    def test_existing_unmanaged_directory_and_active_coordinator_are_rejected(self):
        self.directory.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "no study ledger"):
            self.run_study()
        ledger = study.make_ledger(self.plan, self.directory)
        study.save(self.directory, ledger)
        with (self.directory / "study.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "active local coordinator"):
                self.run_study()
        self.dispatch.assert_not_called()

    def test_invalid_plan_fields_rejected_before_output_creation(self):
        for key, value in (("fractions", [0]), ("fractions", [True]), ("fractions", [.5, .5]),
                           ("seeds", [False]), ("seeds", [2026, 2026]), ("method", "RBD_SCM")):
            invalid = copy.deepcopy(self.plan)
            invalid[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                study.validate_plan(invalid)
        invalid = copy.deepcopy(self.plan)
        invalid["epochs"] = 10
        with self.assertRaises(ValueError):
            study.validate_plan(invalid)
        invalid = copy.deepcopy(self.plan)
        invalid["base_config"]["training"]["train_fraction"] = .5
        with self.assertRaises(ValueError):
            study.validate_plan(invalid)


class PublicStudyAndDispatchTests(unittest.TestCase):
    def test_study_routes_without_loading_a_default_toml(self):
        with mock.patch.object(study_remote, "run_study", return_value=0) as run, mock.patch.object(remote, "load_config") as load:
            self.assertEqual(remote.main(["study", "--plan", "/tmp/plan.json", "--study-dir", "/tmp/study"]), 0)
        run.assert_called_once_with(Path("/tmp/plan.json"), study_dir=Path("/tmp/study"),
                                    dry_run=False, kind="scale")
        load.assert_not_called()

    def test_study_only_options_require_study_and_do_not_reach_other_packs(self):
        from scripts.base_pack.localization_networks.workflow import remote as base
        from scripts.joint_feature_pack.localization_networks.workflow import remote as joint
        for parser, args in ((remote.make_parser(), ["study"]),
                             (remote.make_parser(), ["study", "--plan", "p", "--preset", "full"]),
                             (remote.make_parser(), ["--plan", "p"]),
                             (base.make_parser(), ["study", "--plan", "p"]),
                             (joint.make_parser(), ["study", "--plan", "p"])):
            with self.subTest(args=args), mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(args)

    def test_source_mismatch_is_rejected_before_remote_initialization(self):
        # Restore real dispatch even when invoked from a study's mocked tests.
        config = remote.load_config(remote.DEFAULT_CONFIG)
        run_id = make_run_id(study.PACK, config)
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(remote, "LOCAL_RUNS", Path(temporary)), \
                mock.patch.object(remote, "freeze_source", return_value={"source_sha256": "a" * 64}), \
                mock.patch.object(remote, "remote_python") as ssh, \
                mock.patch("sys.stdout", io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                remote.dispatch(config, "experiment", False, run_id=run_id, expected_source_sha256="b" * 64)
        ssh.assert_not_called()

    def test_study_follow_bounds_transport_failures(self):
        def fake_follow(run_id, *, progress, fetch, summarize):
            for _ in range(3):
                try:
                    progress({})
                except remote.RemoteConnectionError:
                    continue
        with mock.patch.object(remote, "follow", side_effect=fake_follow), \
                mock.patch.object(remote, "progress", side_effect=remote.RemoteConnectionError("offline")) as probe:
            with self.assertRaisesRegex(RuntimeError, "failed 3 consecutive"):
                remote.follow_run({}, "fixture", max_connection_failures=3)
        self.assertEqual(probe.call_count, 3)


if __name__ == "__main__":
    unittest.main()
