"""Durable remote scheduling contracts; filesystem fixtures never launch training."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import fcntl
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.MoE_data_scale.localization_networks.workflow import remote, study
from scripts.MoE_data_scale.localization_networks.workflow import study_worker as worker


class RemoteStudyWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swellex-remote-study-")
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.root = self.workspace / "studies" / "fixture-study"
        self.root.mkdir(parents=True)
        self.ledger = {
            "schema_version": 1, "pack": remote.PACK,
            "study_id": self.root.name, "status": "INTERRUPTED", "updated_at": "earlier",
            "source_sha256": "a" * 64, "source_files": [],
            "plan": {"base_config": {"remote": {"workspace": str(self.workspace)}}},
            "children": [self.child(1)],
        }
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.validate = stack.enter_context(mock.patch.object(worker, "validate"))
        self.verify = stack.enter_context(mock.patch.object(worker, "verify_completed"))
        self.launch = stack.enter_context(mock.patch.object(
            worker.subprocess, "Popen", side_effect=AssertionError("Tests must not launch processes")))
        stack.enter_context(mock.patch("sys.stdout", io.StringIO()))

    def execute(self):
        with mock.patch.object(worker, "process_identity", return_value={
                "pid": 12345, "boot_id": "fixture", "start_ticks": "10", "state": "R"}):
            return worker.execute(self.root)

    @staticmethod
    def child(number, status="PLANNED"):
        result = {"run_id": f"20261015T154330Z-moe-full-RBD_SCM-complex_MoE_e1-{number:08x}",
                  "seed": 2029, "train_fraction": .125 * number,
                  "experts": 1, "network": "complex_MoE_e1", "model": "rbd_scm_complex_moe_e1_range", "config_sha256": str(number) * 64,
                  "status": status}
        if status != "PLANNED":
            result["attempted_at"] = "earlier-attempt"
        return result

    def save_ledger(self):
        remote.dump(self.root / "study.json", self.ledger)

    def report(self, child, status, checks=()):
        run = self.workspace / "runs" / child["run_id"]
        remote.dump(run / "report.json", {"pack": remote.PACK, "run_id": child["run_id"],
                                          "status": status, "checks": list(checks)})
        return run

    def saved(self):
        return study.read_json(self.root / "study.json")

    def test_completed_results_are_preserved_and_only_planned_child_is_launched(self):
        completed = self.child(1, "PASS")
        pending = self.child(2)
        self.ledger["children"] = [completed, pending]
        prior = self.report(completed, "PASS")
        (prior / "checkpoint.pt").write_bytes(b"completed checkpoint must stay unchanged")
        before = {p.name: p.read_bytes() for p in prior.iterdir()}
        self.save_ledger()

        def dispatch(root, ledger, child):
            self.assertEqual(child["run_id"], pending["run_id"])
            child.update(status="DISPATCHING", attempted_at="new-attempt")
            self.report(child, "PASS")

        with mock.patch.object(worker, "prepare_child", side_effect=dispatch) as prepare:
            self.assertEqual(self.execute(), 0)
            prepare.assert_called_once()
        self.assertEqual(self.saved()["status"], "PASS")
        self.assertEqual({p.name: p.read_bytes() for p in prior.iterdir()}, before)
        self.assertEqual([call.args[1]["run_id"] for call in self.verify.call_args_list],
                         [completed["run_id"], pending["run_id"]])

        # Re-entering a completed matrix verifies evidence but launches nothing.
        with mock.patch.object(worker, "prepare_child") as prepare:
            self.assertEqual(self.execute(), 0)
            prepare.assert_not_called()
        self.launch.assert_not_called()

    def test_preflight_rejects_bad_completed_evidence_before_launching_remaining_children(self):
        self.ledger["children"] = [self.child(1, "PASS"), self.child(2)]
        self.save_ledger()
        self.verify.side_effect = ValueError("Completed checkpoint differs")
        with mock.patch.object(worker, "prepare_child") as prepare:
            self.assertEqual(self.execute(), 1)
            prepare.assert_not_called()
        self.assertEqual(self.saved()["status"], "INTERRUPTED")
        self.assertIn("Completed checkpoint differs", self.saved()["error"])

    def test_existing_live_child_is_followed_without_redispatch(self):
        child = self.child(1, "RUNNING")
        self.ledger["children"] = [child]
        self.report(child, "RUNNING")
        self.save_ledger()
        with mock.patch.object(worker, "alive", return_value=True), \
                mock.patch.object(worker, "prepare_child") as prepare, \
                mock.patch.object(worker.time, "sleep", side_effect=lambda _: self.report(child, "PASS")) as wait:
            self.assertEqual(self.execute(), 0)
            wait.assert_called_once()
            prepare.assert_not_called()
        self.assertEqual(self.saved()["children"][0]["run_id"], child["run_id"])
        self.assertEqual(self.saved()["children"][0]["attempted_at"], "earlier-attempt")
        self.verify.assert_called_once()
        self.launch.assert_not_called()

    def test_planned_child_with_existing_directory_is_never_launched(self):
        child = self.ledger["children"][0]
        self.report(child, "RUNNING")
        self.save_ledger()
        with mock.patch.object(worker, "alive", return_value=True), \
                mock.patch.object(worker, "prepare_child") as prepare:
            self.assertEqual(self.execute(), 1)
            prepare.assert_not_called()
        self.assertIn("already exists", self.saved()["error"])
        self.assertEqual(self.saved()["status"], "INTERRUPTED")

    def test_attempted_child_with_missing_directory_is_never_retrained(self):
        self.ledger["children"] = [self.child(1, "INTERRUPTED")]
        self.save_ledger()
        with mock.patch.object(worker, "prepare_child") as prepare:
            self.assertEqual(self.execute(), 1)
            prepare.assert_not_called()
        self.assertIn("directory absent", self.saved()["error"])
        self.assertEqual(self.saved()["children"][0]["attempted_at"], "earlier-attempt")
        self.launch.assert_not_called()

    def test_dead_child_is_not_followed_or_relaunched(self):
        child = self.child(1, "RUNNING")
        self.ledger["children"] = [child]
        self.report(child, "RUNNING")
        self.save_ledger()
        with mock.patch.object(worker, "alive", return_value=False), \
                mock.patch.object(worker, "prepare_child") as prepare, \
                mock.patch.object(worker.time, "sleep") as wait:
            self.assertEqual(self.execute(), 1)
            prepare.assert_not_called()
            wait.assert_not_called()
        self.assertIn("no live launch identity", self.saved()["error"])

    def test_locked_controller_without_launch_marker_cannot_start_a_second_process(self):
        self.save_ledger()
        self.assertFalse((self.root / "controller-launch.json").exists())
        # A separate open-file description emulates the already-running controller.
        with (self.root / "study.lock").open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "locked by a live coordinator"):
                worker.start_locked(self.root)
        self.launch.assert_not_called()
        self.assertEqual(self.saved(), self.ledger)
        self.assertFalse((self.root / "controller-launch.json").exists())

    def test_registered_recovery_rejects_self_consistent_source_replaced_since_original_ledger(self):
        self.save_ledger()
        stage = self.workspace / "study-staging" / "fixture-controller"
        stage_project = stage / "project"
        stage_project.mkdir(parents=True)
        source = self.root / "training-source"
        source.mkdir()
        raw = b"internally consistent replacement lock\n"
        files = [{"path": "uv.lock", "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]
        manifest = {"pack": remote.PACK, "files": files, "source_sha256": study.digest(files)}
        (source / "uv.lock").write_bytes(raw)
        (stage_project / "uv.lock").write_bytes(raw)
        remote.dump(self.root / "training-manifest.json", manifest)
        remote.dump(self.root / "control.json", {"pack": remote.PACK,
                    "study_identity": worker.identity(self.ledger), "controller_source_sha256": "b" * 64})
        # This passes ordinary byte/inventory checks; only comparison with the frozen
        # ledger can detect a coherent replacement of both source and manifest.
        worker.verify_snapshot(source, manifest)
        before = (self.root / "study.json").read_bytes()
        with mock.patch.object(worker, "alive", return_value=False), \
                self.assertRaisesRegex(ValueError, "original study source"):
            worker.install(stage, {"ledger": copy.deepcopy(self.ledger), "controller_manifest": manifest})
        self.assertEqual((self.root / "study.json").read_bytes(), before)
        self.assertNotIn("attempted_at", self.saved()["children"][0])
        self.assertFalse((self.workspace / "runs" / self.ledger["children"][0]["run_id"]).exists())
        self.launch.assert_not_called()

    def test_failed_child_marks_study_failed_and_preserves_failure_reason(self):
        child = self.child(1, "RUNNING")
        self.ledger["children"] = [child]
        run = self.report(child, "FAIL")
        report = study.read_json(run / "report.json")
        report["error"] = "CUDA execution failed in training phase"
        remote.dump(run / "report.json", report)
        before = (run / "report.json").read_bytes()
        self.save_ledger()
        with mock.patch.object(worker, "prepare_child") as prepare, \
                mock.patch.object(worker.time, "sleep") as wait:
            self.assertEqual(self.execute(), 1)
            prepare.assert_not_called()
            wait.assert_not_called()
        self.assertEqual(self.saved()["status"], "FAIL")
        self.assertEqual(self.saved()["children"][0]["status"], "FAIL")
        self.assertIn(child["run_id"], self.saved()["error"])
        self.assertIn(report["error"], self.saved()["error"])
        self.assertEqual((run / "report.json").read_bytes(), before)
        with mock.patch.object(worker, "alive", return_value=False), \
                self.assertRaisesRegex(ValueError, "Failed study"):
            worker.start_locked(self.root)
        self.launch.assert_not_called()

    def test_liveness_requires_same_boot_and_process_start_and_rejects_zombies(self):
        saved = {"pid": 1234, "boot_id": "boot-one", "start_ticks": "5678", "state": "S"}
        marker = self.root / "controller-launch.json"
        remote.dump(marker, saved)
        with mock.patch.object(worker, "process_identity", return_value={**saved, "state": "R"}):
            self.assertTrue(worker.alive(marker))
        for change in ({"boot_id": "boot-two"}, {"start_ticks": "9999"},
                       {"pid": 9999}, {"state": "Z"}):
            with self.subTest(change=change), \
                    mock.patch.object(worker, "process_identity", return_value={**saved, **change}):
                self.assertFalse(worker.alive(marker))
        with mock.patch.object(worker, "process_identity", side_effect=FileNotFoundError):
            self.assertFalse(worker.alive(marker))
        self.assertFalse(worker.alive(self.root / "missing-marker.json"))

    def test_process_stat_comm_with_spaces_and_parentheses_is_parsed(self):
        stat = "1234 (worker (nested) name) S " + " ".join(["0"] * 18 + ["5678", "0", "0"])

        def read(path, *args, **kwargs):
            return "boot-one\n" if path.name == "boot_id" else stat

        with mock.patch.object(Path, "read_text", autospec=True, side_effect=read):
            self.assertEqual(worker.process_identity(1234),
                             {"pid": 1234, "boot_id": "boot-one", "start_ticks": "5678", "state": "S"})

    def test_inspect_is_read_only_and_uses_reports_not_stale_ledger_counts(self):
        self.ledger.update(status="RUNNING", children=[self.child(1, "RUNNING"), self.child(2, "RUNNING")])
        self.save_ledger()
        remote.dump(self.root / "control.json", {"pack": remote.PACK,
                    "study_identity": worker.identity(self.ledger), "controller_source_sha256": "b" * 64})
        self.report(self.ledger["children"][0], "PASS")
        active = self.report(self.ledger["children"][1], "RUNNING", [
            {"name": "evaluate_small", "status": "PASS"},
            {"name": "evaluate_medium", "status": "PASS"},
            {"name": "evaluate_large", "status": "RUNNING"},
            {"name": "train_small", "status": "PASS"}])
        before = {p.relative_to(self.workspace): p.read_bytes() for p in self.workspace.rglob("*") if p.is_file()}
        with mock.patch.object(worker, "alive", side_effect=lambda marker: marker == active / "launch.json"), \
                mock.patch.object(worker, "suite_selection", return_value=("selected", {"method": "RBD_SCM", "network": "complex_MoE_e1"})), \
                mock.patch.object(remote, "model_plan", return_value=[{}]):
            result = worker.inspect(self.root)
        self.assertEqual(result["status"], "INTERRUPTED")
        self.assertFalse(result["controller_alive"])
        self.assertEqual((result["completed_children"], result["completed_models"], result["total_models"]), (1, 3, 2))
        self.assertEqual([c["completed_models"] for c in result["children"]], [1, 2])
        after = {p.relative_to(self.workspace): p.read_bytes() for p in self.workspace.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.launch.assert_not_called()

    def test_child_directory_without_report_is_unknown_and_never_reinitialized(self):
        child = self.child(1, "DISPATCHING")
        run = self.workspace / "runs" / child["run_id"]
        run.mkdir(parents=True)
        self.assertFalse((run / "report.json").exists())
        before = sorted(p.relative_to(self.workspace).as_posix() for p in self.workspace.rglob("*"))
        result = worker.child_status(self.workspace, child["run_id"])
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["run_id"], child["run_id"])
        self.assertIn("initialization has no report", result["error"])
        self.assertEqual(sorted(p.relative_to(self.workspace).as_posix() for p in self.workspace.rglob("*")), before)
        self.launch.assert_not_called()

    def test_inspect_reports_initialization_window_without_writing_or_relaunching(self):
        child = self.child(1, "DISPATCHING")
        self.ledger.update(status="RUNNING", children=[child])
        self.save_ledger()
        run = self.workspace / "runs" / child["run_id"]
        run.mkdir(parents=True)
        remote.dump(self.root / "control.json", {"pack": remote.PACK,
                    "study_identity": worker.identity(self.ledger), "controller_source_sha256": "b" * 64})
        before_files = {p.relative_to(self.workspace): p.read_bytes()
                        for p in self.workspace.rglob("*") if p.is_file()}
        before_paths = sorted(p.relative_to(self.workspace).as_posix() for p in self.workspace.rglob("*"))
        for running in (True, False):
            with self.subTest(controller_alive=running), \
                    mock.patch.object(worker, "alive", side_effect=lambda marker: running and marker == self.root / "controller-launch.json"), \
                    mock.patch.object(worker, "suite_selection", return_value=("selected", {"method": "RBD_SCM", "network": "complex_MoE_e1"})), \
                    mock.patch.object(remote, "model_plan", return_value=[{}]):
                result = worker.inspect(self.root)
            self.assertEqual(result["status"], "RUNNING" if running else "INTERRUPTED")
            self.assertEqual(result["controller_alive"], running)
            self.assertEqual(result["children"][0]["status"], "DISPATCHING" if running else "UNKNOWN")
            self.assertEqual((result["completed_children"], result["completed_models"], result["total_models"]), (0, 0, 1))
        self.assertEqual({p.relative_to(self.workspace): p.read_bytes()
                          for p in self.workspace.rglob("*") if p.is_file()}, before_files)
        self.assertEqual(sorted(p.relative_to(self.workspace).as_posix() for p in self.workspace.rglob("*")), before_paths)
        self.launch.assert_not_called()

    def test_identity_ignores_progress_but_detects_frozen_source_plan_and_child_changes(self):
        initial = worker.identity(self.ledger)
        changed = copy.deepcopy(self.ledger)
        changed.update(status="PASS", error="transient", updated_at="later")
        changed["children"][0].update(status="PASS", error="transient", attempted_at="later")
        self.assertEqual(worker.identity(changed), initial)
        variants = []
        source = copy.deepcopy(changed)
        source["source_sha256"] = "b" * 64
        variants.append(source)
        plan = copy.deepcopy(changed)
        plan["plan"]["epoch_budget"] = 200
        variants.append(plan)
        child = copy.deepcopy(changed)
        child["children"][0]["run_id"] = self.child(2)["run_id"]
        variants.append(child)
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(worker.identity(variant), initial)


class FrozenStudySourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swellex-study-source-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.contents = {"uv.lock": b"frozen lock\n", "scripts/train.py": b"print('frozen source')\n"}
        for name, raw in self.contents.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        files = [{"path": name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                 for name, raw in sorted(self.contents.items())]
        self.manifest = {"files": files, "source_sha256": study.digest(files)}

    def test_copy_includes_only_manifest_sources_and_never_training_outputs(self):
        for name in ("outputs/model.pt", "output/report.json", "Origindata/raw.mat", "__pycache__/cached.pyc"):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"must not be copied")
        target = self.root / "copied"
        worker.copy_snapshot(self.source, target, self.manifest)
        self.assertEqual({p.relative_to(target).as_posix(): p.read_bytes()
                          for p in target.rglob("*") if p.is_file()}, self.contents)

    def test_same_size_source_corruption_and_manifest_digest_drift_are_rejected(self):
        path = self.source / "uv.lock"
        path.write_bytes(b"broken lock\n")  # Same length: size alone must not authorize reuse.
        with self.assertRaisesRegex(ValueError, "Frozen source differs"):
            worker.copy_snapshot(self.source, self.root / "bad-bytes", self.manifest)
        path.write_bytes(self.contents["uv.lock"])
        wrong = {**self.manifest, "source_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "manifest digest"):
            worker.copy_snapshot(self.source, self.root / "bad-manifest", wrong)

    def test_external_symlinks_and_parent_path_escape_are_rejected(self):
        outside = self.root / "outside.py"
        outside.write_bytes(self.contents["scripts/train.py"])
        path = self.source / "scripts/train.py"
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Frozen source differs"):
            worker.copy_snapshot(self.source, self.root / "linked", self.manifest)
        escaped = {"files": [{"path": "../outside.py", "size": outside.stat().st_size,
                               "sha256": remote.sha256(outside)}], "source_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "Unsafe frozen source path"):
            worker.copy_snapshot(self.source, self.root / "escaped", escaped)


if __name__ == "__main__":
    unittest.main()
