"""Stdlib contracts for dispatch, source integrity, and prediction reports.

Execute on the compute host with unittest discovery; no numerical imports or SSH
are needed. Temporary Git repositories model the actual working-tree snapshot.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import csv
import hashlib
import json
import io
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

WORKFLOW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW))
import config
import remote
import worker

RUN_ID = "20261008T120000Z-01234567"
FIELDS = ["sample", "source_segment_idx", "window_center_s", "pred_range_km", "true_range_km", "abs_error_km"]


def valid_config():
    return copy.deepcopy(config.load_config(WORKFLOW / "haiqin.toml"))


class ConfigurationTests(unittest.TestCase):
    def test_current_configuration_and_unknown_fields(self):
        original = valid_config()
        self.assertEqual(config.validate_config(copy.deepcopy(original)), original)
        for mutate in (
            lambda c: c.update(unknown={}),
            lambda c: c["dataset"].update(unknown=1),
            lambda c: c.pop("training"),
            lambda c: c["validation"].pop("seed"),
        ):
            with self.subTest(mutation=mutate):
                candidate = copy.deepcopy(original)
                mutate(candidate)
                with self.assertRaises(ValueError):
                    config.validate_config(candidate)

    def test_invalid_numeric_and_feature_contracts(self):
        cases = [
            ("schema_version", None, True),
            ("validation", "threads", 0),
            ("validation", "seed", True),
            ("validation", "device", "cuda; echo unsafe"),
            ("training", "epochs", 1.5),
            ("training", "huber_beta", float("nan")),
            ("training", "val_fraction", 1.0),
            ("dataset", "frequency_selection_modes", ["deep", "deep"]),
            ("dataset", "frequency_selection_modes", ["full", "deep"]),
            ("dataset", "manual_dataset_variant_tag", "../escape"),
            ("dataset", "snapshot_overlap_count", 1),
            ("grnn", "cv_folds", 1),
            ("grnn", "sigma_candidates", [0.0]),
        ]
        for table, key, value in cases:
            with self.subTest(table=table, key=key, value=value):
                candidate = valid_config()
                if key is None:
                    candidate[table] = value
                else:
                    candidate[table][key] = value
                with self.assertRaises(ValueError):
                    config.validate_config(candidate)

    def test_dangerous_remote_paths_and_hosts(self):
        baseline = valid_config()
        repo = baseline["remote"]["repository"]
        values = [("host", "-oProxyCommand=unsafe"), ("host", "haiqin; echo unsafe")]
        values += [("workspace", p) for p in ("/", repo, str(Path(repo).parent), repo + "/runs", "/mnt/../escape", "relative/path", "/tmp/work space")]
        values += [("base_python", "/usr/bin/python$(echo unsafe)"), ("uv", "/usr/local/bin/../uv")]
        for key, value in values:
            with self.subTest(key=key, value=value):
                candidate = copy.deepcopy(baseline)
                candidate["remote"][key] = value
                with self.assertRaises(ValueError):
                    config.validate_config(candidate)

    def test_run_id_cannot_escape_workspace(self):
        cfg = valid_config()
        self.assertTrue(remote.run_dir(cfg, RUN_ID).endswith("/runs/" + RUN_ID))
        for run in ("../escape", "/tmp/escape", RUN_ID + "/../escape", "20261008T120000Z-$(id)"):
            with self.subTest(run=run), self.assertRaises(ValueError):
                remote.run_dir(cfg, run)

    def test_obsolete_remote_and_timeout_fields_are_rejected(self):
        for table, field in (("remote", "python"), ("remote", "matlab"),
                             ("remote", "matlab_preload"), ("remote", "matlab_preload_sha256"),
                             ("validation", "matlab_timeout_s")):
            candidate = valid_config()
            candidate[table][field] = "obsolete"
            with self.subTest(table=table, field=field), self.assertRaises(ValueError):
                config.validate_config(candidate)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="swellex-workflow-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "source"
        self.project.mkdir()
        self.git("init", "-q")
        self.put(".gitignore", "__pycache__/\n*.pyc\n")
        self.put("README.md", "committed documentation\n")
        self.put("pyproject.toml", "[project]\nname='fixture'\n")
        self.put("uv.lock", "version=1\n")
        self.put(".python-version", "3.14.4\n")
        self.put("scripts/localization_networks/workflow/worker.py", "# committed worker\n")
        self.put("scripts/data_processing/matlab_reference/deleted.m", "% removed after commit\n")
        self.put("outputs/large.h5", "generated data must stay local\n")
        self.put("output/data/run/report.json", "new result layout must stay local\n")
        self.git("add", ".")
        self.git("-c", "user.name=Workflow Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
        self.put("README.md", "uncommitted current documentation\n")
        self.put("scripts/localization_networks/new_module.py", "VALUE = 'new source'\n")
        self.put("scripts/localization_networks/__pycache__/module.pyc", "ignored cache\n")
        self.put("scripts/localization_networks/.DS_Store", "OS junk\n")
        self.put("scripts/data_processing/matlab_reference/._sidecar.m", "OS sidecar\n")
        self.put("scripts/data_processing/matlab_reference/scratch.asv", "MATLAB autosave\n")
        (self.project / "scripts/data_processing/matlab_reference/deleted.m").unlink()

    def git(self, *arguments):
        return subprocess.run(["git", *arguments], cwd=self.project, check=True, capture_output=True, text=True)

    def put(self, name, content):
        path = self.project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def freeze(self):
        stage = self.root / "snapshot"
        stage.mkdir()
        manifest = remote.freeze_source(stage, valid_config(), RUN_ID, "quick", self.project)
        return stage, manifest

    def test_working_tree_bytes_new_files_and_deleted_files(self):
        stage, manifest = self.freeze()
        names = {item["path"] for item in manifest["files"]}
        self.assertIn("scripts/localization_networks/new_module.py", names)
        self.assertTrue({"pyproject.toml", "uv.lock", ".python-version"}.issubset(names))
        self.assertNotIn("scripts/data_processing/matlab_reference/deleted.m", names)
        self.assertNotIn("outputs/large.h5", names)
        self.assertNotIn("output/data/run/report.json", names)
        self.assertFalse(any("__pycache__" in name or "._" in name or name.endswith((".asv", ".DS_Store")) for name in names))
        self.assertEqual((stage / "README.md").read_text(), "uncommitted current documentation\n")
        worker.verify_snapshot(stage, manifest)
        self.put("README.md", "later source edit\n")
        worker.verify_snapshot(stage, manifest)
        self.assertEqual((stage / "README.md").read_text(), "uncommitted current documentation\n")

    def test_generated_data_and_figure_do_not_change_source_inventory(self):
        stage, manifest = self.freeze()
        for relative in ("output/data/run/report.json", "output/figure/comparison.png", "outputs/networks_results/history.json"):
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("generated result")
        worker.verify_snapshot(stage, manifest)

    def test_changed_source_bytes_and_manifest_digest_rejected(self):
        stage, manifest = self.freeze()
        original = (stage / "README.md").read_bytes()
        # Same-sized corruption exercises the hash check rather than size only.
        (stage / "README.md").write_bytes(b"X" + original[1:])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            worker.verify_snapshot(stage, manifest)
        (stage / "README.md").write_bytes(original)
        damaged = copy.deepcopy(manifest)
        damaged["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            worker.verify_snapshot(stage, damaged)

    def test_extra_inventory_and_missing_file_rejected(self):
        stage, manifest = self.freeze()
        extra = stage / "scripts/localization_networks/injected.py"
        extra.write_text("# unexpected source\n")
        with self.assertRaisesRegex(ValueError, "inventory"):
            worker.verify_snapshot(stage, manifest)
        extra.unlink()
        (stage / "README.md").unlink()
        with self.assertRaises(ValueError):
            worker.verify_snapshot(stage, manifest)

    def test_external_source_symlink_rejected(self):
        external = self.root / "external.py"
        external.write_text("# external source\n")
        (self.project / "scripts/localization_networks/external.py").symlink_to(external)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            remote.source_paths(self.project)

    def test_manifest_traversal_and_duplicates_rejected(self):
        stage, manifest = self.freeze()
        for mutate in (
            lambda m: m["files"].append(copy.deepcopy(m["files"][0])),
            lambda m: m["files"][0].update(path="../escape"),
            lambda m: m["files"][0].update(path="/tmp/escape"),
        ):
            candidate = copy.deepcopy(manifest)
            mutate(candidate)
            candidate["source_sha256"] = hashlib.sha256(json.dumps(candidate["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            with self.assertRaises(ValueError):
                worker.verify_snapshot(stage, candidate)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="swellex-prediction-test-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "predictions.csv"
        self.rows = [dict(zip(FIELDS, row)) for row in (
            (0, 3, 2.5, 2.0, 1.0, 1.0),
            (1, 8, 7.5, 2.0, 4.0, 2.0),
        )]

    def save(self, rows, fields=FIELDS):
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_metrics(self):
        self.save(self.rows)
        result = worker.prediction_metrics(self.path)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["source_segment_indices"], [3, 8])
        self.assertEqual(result["mae_km"], 1.5)
        self.assertAlmostEqual(result["rmse_km"], (2.5 ** 0.5))
        self.assertEqual(result["max_error_window_center_s"], 7.5)

    def test_empty_missing_and_malformed_fields_rejected(self):
        self.save([])
        with self.assertRaises(ValueError):
            worker.prediction_metrics(self.path)
        self.save(self.rows, FIELDS[:-1])
        with self.assertRaises((ValueError, KeyError)):
            worker.prediction_metrics(self.path)
        for key, value in (("pred_range_km", "bad-number"), ("source_segment_idx", "1.5"), ("abs_error_km", "wrong")):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            self.save(rows)
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.prediction_metrics(self.path)

    def test_nonfinite_columns_rejected(self):
        for key in ("pred_range_km", "true_range_km", "window_center_s", "abs_error_km"):
            for value in ("nan", "inf", "-inf"):
                rows = copy.deepcopy(self.rows)
                rows[0][key] = value
                self.save(rows)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    worker.prediction_metrics(self.path)

    def test_misaligned_rows_and_error_columns_rejected(self):
        cases = [(0, "abs_error_km", 9), (1, "sample", 0), (0, "sample", 1),
                 (1, "source_segment_idx", 3), (0, "source_segment_idx", 0),
                 (1, "source_segment_idx", 2)]
        for index, key, value in cases:
            rows = copy.deepcopy(self.rows)
            rows[index][key] = value
            self.save(rows)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                worker.prediction_metrics(self.path)

    def test_csv_must_match_expected_test_windows_labels_and_count(self):
        expected = [(3, 2.5, 1.0), (8, 7.5, 4.0)]
        self.save(self.rows)
        self.assertEqual(worker.prediction_metrics(self.path, expected)["samples"], 2)
        # Each corruption remains internally consistent and can only be rejected
        # by comparison against the independently loaded test dataset.
        mismatches = [expected[:1], [(4, 2.5, 1.0), expected[1]],
                      [(3, 2.6, 1.0), expected[1]], [(3, 2.5, 1.1), expected[1]]]
        for windows in mismatches:
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                worker.prediction_metrics(self.path, windows)


class QuotingTests(unittest.TestCase):
    def test_shell_round_trip_preserves_code_and_arguments(self):
        with tempfile.TemporaryDirectory(prefix="swellex-quote-test-") as directory:
            marker = Path(directory) / "injection-was-executed"
            values = ["two words", "single'quote", 'double"quote', f"$(touch {marker})", "; echo injection", "line\nnewline", "\\"]
            argv = [sys.executable, "-c", "import json,sys;print(json.dumps(sys.argv[1:]))", *values]
            command = shlex.join(argv)
            self.assertEqual(shlex.split(command), argv)
            result = subprocess.run(["/bin/sh", "-c", command], check=True, text=True, capture_output=True)
            self.assertEqual(json.loads(result.stdout), values)
            self.assertFalse(marker.exists())

    def test_ssh_command_quotes_python_code_and_passes_json_on_stdin(self):
        cfg = valid_config()
        code = "print('quote'); value = '$HOME'; other = '\\n'"
        payload = {"name": "spaces and 'quotes'", "number": 4}
        completed = subprocess.CompletedProcess([], 0, stdout='{"status":"PASS"}\n', stderr="")
        with mock.patch.object(remote.subprocess, "run", return_value=completed) as run:
            self.assertEqual(remote.remote_python(cfg, code, payload), {"status": "PASS"})
        positional, keyword = run.call_args
        argv = positional[0]
        self.assertEqual(argv[0], "ssh")
        self.assertEqual(argv[-2], cfg["remote"]["host"])
        self.assertEqual(shlex.split(argv[-1]), [cfg["remote"]["base_python"], "-c", code])
        self.assertEqual(json.loads(keyword["input"]), payload)



class LifecycleTests(unittest.TestCase):
    def test_early_failure_persists_terminal_report_and_exports_evidence(self):
        with tempfile.TemporaryDirectory(prefix="swellex-failure-test-") as directory:
            run = Path(directory)
            instance = worker.Worker(run)
            with mock.patch.object(instance, "setup", side_effect=ValueError("damaged snapshot")):
                self.assertEqual(instance.execute(), 1)
            report = json.loads((run / "report.json").read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["exit_code"], 1)
            self.assertEqual(report["checks"][0]["status"], "FAIL")
            self.assertIn("damaged snapshot", report["error"])
            self.assertEqual(json.loads((run / "export/data/report.json").read_text()), report)
            self.assertTrue((run / "export/figure").is_dir())

    def test_status_distinguishes_running_worker_from_stale_pid_identity(self):
        # Remote STATUS_CODE is deliberately stdlib-only and can run without
        # loading a project or importing any numerical dependency.
        with tempfile.TemporaryDirectory(prefix="swellex-status-test-") as directory:
            run = Path(directory)
            pid = os.getpid()
            identity = {"pid": pid, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                        "start_ticks": Path(f"/proc/{pid}/stat").read_text().split()[21]}
            launch = run / "launch.json"
            launch.write_text(json.dumps(identity))
            def query():
                proc = subprocess.run([sys.executable, "-c", remote.STATUS_CODE], input=json.dumps({"run_dir": str(run)}),
                                      text=True, capture_output=True, check=True)
                return json.loads(proc.stdout)
            self.assertEqual(query()["status"], "STARTING")
            identity["start_ticks"] = "-1"
            launch.write_text(json.dumps(identity))
            result = query()
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["exit_code"], 1)
            terminal = {"run_id": RUN_ID, "status": "PASS", "exit_code": 0, "checks": []}
            (run / "report.json").write_text(json.dumps(terminal))
            self.assertEqual(query(), terminal)


class ResultDeliveryTests(unittest.TestCase):
    def test_visualization_manifest_is_retained_and_render_failure_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="swellex-render-phase-test-") as directory:
            instance = worker.Worker(Path(directory))
            manifest = {"status": "PASS", "figures": ["comparison.png"]}
            render = mock.Mock(return_value=manifest)
            module = mock.Mock(render_run=render)
            original_path = sys.path.copy()
            try:
                with mock.patch.dict(sys.modules, {"scripts.visualization.plot_results": module}):
                    self.assertEqual(instance.visualize(), manifest)
                    render.assert_called_once_with(instance.run, instance.project / "output/figure", require_pass=False)
                    self.assertEqual(json.loads((instance.run / "reports/visualization.json").read_text()), manifest)
                    render.return_value = {"status": "FAIL"}
                    with self.assertRaisesRegex(ValueError, "PASS manifest"):
                        instance.visualize()
            finally:
                sys.path[:] = original_path

    def test_local_result_roots_are_singular_output(self):
        self.assertEqual(remote.LOCAL_RUNS, remote.PROJECT / "output/data")
        self.assertEqual(remote.LOCAL_FIGURES, remote.PROJECT / "output/figure")

    def test_export_separates_logs_history_predictions_and_figures(self):
        with tempfile.TemporaryDirectory(prefix="swellex-export-test-") as directory:
            run = Path(directory)
            content = {
                "report.json": '{"status":"PASS"}',
                "logs/train.log": "training log",
                "reports/metrics.json": "{}",
                "project/outputs/networks_results/RBD/complex_cnn_range/train_outputs/data_history_1008_120000.json": "[]",
                "project/outputs/networks_results/RBD/complex_cnn_range/test_outputs/data_predictions_1008_120000.csv": "sample,pred_range_km",
                "project/outputs/networks_results/RBD/complex_cnn_range/test_outputs/data_range_prediction_1008_120000.png": "model PNG",
                "project/outputs/networks_results/0_GRNN_related/scm_grnn_range/test_outputs/data_range_prediction_1008_120000.png": "GRNN PNG",
                "project/output/figure/comparison.pdf": "aggregate PDF",
                "project/output/figure/source-report.json": '{"status":"RUNNING"}',
                "project/output/figure/metrics.csv": "method,model,mae_km",
                "project/output/figure/manifest.json": '{"status":"PASS"}',
                "project/output/figure/nested/render.log": "render log",
                "project/outputs/networks_results/RBD/complex_cnn_range/train_outputs/checkpoint.pt": "large checkpoint",
                "project/outputs/Datasets/large.h5": "large data",
            }
            for relative, value in content.items():
                path = run / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            outside = run / "private.csv"
            outside.write_text("external file")
            (run / "logs/external.csv").symlink_to(outside)
            instance = worker.Worker(run)
            instance.export()
            instance.export()  # Re-export is safe and deterministic.
            data = run / "export/data"
            figures = run / "export/figure"
            self.assertEqual((data / "logs/train.log").read_text(), "training log")
            self.assertTrue((data / "project/outputs/networks_results/RBD/complex_cnn_range/train_outputs/data_history_1008_120000.json").is_file())
            self.assertTrue((data / "project/outputs/networks_results/RBD/complex_cnn_range/test_outputs/data_predictions_1008_120000.csv").is_file())
            self.assertEqual((figures / "models/RBD/complex_cnn_range/test_outputs/data_range_prediction_1008_120000.png").read_text(), "model PNG")
            self.assertEqual((figures / "models/SCM/scm_grnn_range/test_outputs/data_range_prediction_1008_120000.png").read_text(), "GRNN PNG")
            self.assertEqual((figures / "comparison.pdf").read_text(), "aggregate PDF")
            for relative in ("source-report.json", "metrics.csv", "manifest.json", "nested/render.log"):
                self.assertEqual((data / "reports/visualization" / relative).read_text(),
                                 content["project/output/figure/" + relative])
                self.assertFalse((figures / relative).exists())
            self.assertFalse(list(data.rglob("*.png")) + list(data.rglob("*.pdf")))
            self.assertFalse(list(figures.rglob("*.json")) + list(figures.rglob("*.csv")))
            self.assertFalse(list((run / "export").rglob("*.pt")) + list((run / "export").rglob("*.h5")))
            self.assertFalse((data / "logs/external.csv").exists())

    def test_fetch_uses_distinct_data_figure_and_live_log_transfers(self):
        with tempfile.TemporaryDirectory(prefix="swellex-fetch-test-") as directory:
            root = Path(directory)
            report = {"run_id": RUN_ID, "status": "PASS", "checks": []}
            completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            with mock.patch.object(remote, "LOCAL_RUNS", root / "data"), mock.patch.object(remote, "LOCAL_FIGURES", root / "figure"), \
                 mock.patch.object(remote, "status", return_value=report), mock.patch.object(remote.subprocess, "run", return_value=completed) as call, \
                 mock.patch("builtins.print"):
                self.assertEqual(remote.fetch(valid_config(), RUN_ID), report)
            arguments = [entry.args[0] for entry in call.call_args_list]
            self.assertEqual(len(arguments), 3)
            self.assertTrue(arguments[0][-2].endswith("/export/data/"))
            self.assertEqual(arguments[0][-1], str(root / "data" / RUN_ID) + "/")
            self.assertTrue(arguments[1][-2].endswith("/export/figure/"))
            self.assertEqual(arguments[1][-1], str(root / "figure" / RUN_ID) + "/")
            self.assertTrue(arguments[2][-2].endswith("/logs/"))
            self.assertEqual(arguments[2][-1], str(root / "data" / RUN_ID / "logs") + "/")
            self.assertEqual(json.loads((root / "data" / RUN_ID / "report.json").read_text()), report)

    def test_initialization_exports_starting_report_to_data_and_creates_figure_directory(self):
        with tempfile.TemporaryDirectory(prefix="swellex-init-export-test-") as directory:
            root = Path(directory)
            (root / "repository/Origindata").mkdir(parents=True)
            payload = {"workspace": str(root / "workspace"), "repository": str(root / "repository"),
                       "run_id": RUN_ID, "dispatch_token": "fixture", "report": {"status": "STARTING"},
                       "manifest.json": {}, "request.json": {}, "generation_config.json": {}}
            proc = subprocess.run([sys.executable, "-c", remote.INIT_CODE], input=json.dumps(payload),
                                  text=True, capture_output=True, check=True)
            run = Path(json.loads(proc.stdout)["run_dir"])
            self.assertEqual(json.loads((run / "export/data/report.json").read_text())["status"], "STARTING")
            self.assertTrue((run / "export/figure").is_dir())
            self.assertFalse((run / "export/report.json").exists())


class UvRuntimeTests(unittest.TestCase):
    def test_launch_uses_lock_specific_environment_and_existing_python(self):
        with tempfile.TemporaryDirectory(prefix="swellex-uv-launch-test-") as directory:
            run = Path(directory) / "run"
            project = run / "project"
            project.mkdir(parents=True)
            (run / "logs").mkdir()
            for name in ("manifest.json", "request.json", "generation_config.json"):
                (project / name).write_text("{}")
            lock_bytes = b"frozen dependencies fixture\n"
            (project / "uv.lock").write_bytes(lock_bytes)
            lock_hash = hashlib.sha256(lock_bytes).hexdigest()
            payload = {"run_dir": str(run), "workspace": str(Path(directory) / "workspace"),
                       "base_python": "/existing/conda/bin/python", "uv": "/isolated/tools/uv"}
            ambient = {"UV_PROJECT_ENVIRONMENT": "/wrong/shared/environment", "UV_PYTHON_DOWNLOADS": "automatic"}
            with mock.patch.dict(os.environ, ambient), \
                 mock.patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), \
                 mock.patch.object(sys, "stdout", io.StringIO()), \
                 mock.patch.object(subprocess, "Popen", return_value=mock.Mock(pid=os.getpid())) as launch:
                before = dict(os.environ)
                scope = {}
                try:
                    exec(remote.START_CODE, scope)
                finally:
                    if "lock" in scope:
                        scope["lock"].close()
                self.assertEqual(dict(os.environ), before)
            argv = launch.call_args.args[0]
            environment = launch.call_args.kwargs["env"]
            self.assertEqual(argv[:7], [payload["uv"], "run", "--frozen", "--python", payload["base_python"], "--project", str(project)])
            self.assertEqual(argv[7:9], ["python", "-B"])
            self.assertEqual(environment["UV_PROJECT_ENVIRONMENT"], str(Path(payload["workspace"]) / "environments" / lock_hash / ".venv"))
            self.assertEqual(environment["UV_CACHE_DIR"], str(Path(payload["workspace"]) / "uv_cache"))
            self.assertEqual(environment["UV_PYTHON_DOWNLOADS"], "never")
            self.assertTrue(launch.call_args.kwargs["start_new_session"])
            self.assertEqual(json.loads((run / "launch.json").read_text())["pid"], os.getpid())

    def test_worker_rejects_a_shared_or_other_lock_environment(self):
        with tempfile.TemporaryDirectory(prefix="swellex-uv-prefix-test-") as directory:
            workspace = Path(directory) / "workspace"
            run = workspace / "runs" / RUN_ID
            project = run / "project"
            project.mkdir(parents=True)
            cfg = valid_config()
            cfg["remote"]["workspace"] = str(workspace)
            cfg["remote"]["repository"] = str(Path(directory) / "source")
            lock_bytes = b"frozen dependencies fixture\n"
            (project / "uv.lock").write_bytes(lock_bytes)
            lock_hash = hashlib.sha256(lock_bytes).hexdigest()
            request = {"schema_version": 1, "run_id": RUN_ID, "suite": "quick", "config": cfg, "use_cache": True}
            manifest = {"run_id": RUN_ID, "suite": "quick", "git_head": "fixture", "files": [], "source_sha256": "a" * 64,
                        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()}
            generation = {"schema_version": 1, "dataset": cfg["dataset"], "rbd": cfg["rbd"], "result_path": "outputs/validation/generation_result.json"}
            for name, value in (("request.json", request), ("manifest.json", manifest), ("generation_config.json", generation)):
                worker.write_json(run / name, value)
            with mock.patch.object(worker, "verify_snapshot"), mock.patch.object(sys, "prefix", str(workspace / ".venv")):
                with self.assertRaisesRegex(ValueError, "uv environment"):
                    worker.Worker(run).setup()
            instance = worker.Worker(run)
            with mock.patch.object(worker, "verify_snapshot"), mock.patch.object(sys, "prefix", str(worker.workspace_environment(cfg, lock_hash))):
                self.assertEqual(instance.setup()["source_sha256"], manifest["source_sha256"])
            self.assertEqual(instance.lock_hash, lock_hash)

    def test_cache_identity_tracks_lock_raw_inputs_and_generator_sources(self):
        with tempfile.TemporaryDirectory(prefix="swellex-uv-cache-test-") as directory:
            run = Path(directory)
            instance = worker.Worker(run)
            instance.config = valid_config()
            repository = run / "raw_repository"
            raw = repository / "Origindata"
            names = ["positions/positions_vla.txt", "events/range/RangeEventS5/SproulToVLA.S5.txt", "environments/ctds/i9605.prn"]
            names += [f"events/S5/vla_matfiles/S5_VLA_NO_{n:02d}.mat" for n in range(1, 22)]
            for name in names:
                path = raw / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"raw input identity fixture")
            instance.project.mkdir()
            (instance.project / "Origindata").symlink_to(raw, target_is_directory=True)
            instance.config["remote"]["repository"] = str(repository)
            instance.config["remote"]["workspace"] = str(run / "workspace")
            instance.lock_hash = "a" * 64
            instance.manifest = {"files": [{"path": "scripts/data_processing/generate.py", "sha256": "b" * 64},
                                            {"path": "scripts/data_processing/matlab_reference/reference.m", "sha256": "c" * 64}]}
            with mock.patch.object(worker.importlib.metadata, "version", return_value="pinned-fixture"):
                first = instance.data_identity()["cache_key"]
                self.assertEqual(instance.data_identity()["cache_key"], first)
                instance.lock_hash = "d" * 64
                changed_lock = instance.data_identity()["cache_key"]
                self.assertNotEqual(changed_lock, first)
                self.assertEqual(worker.workspace_environment(instance.config, instance.lock_hash), run / "workspace/environments" / instance.lock_hash / ".venv")
                instance.manifest["files"][1]["sha256"] = "e" * 64
                self.assertEqual(instance.data_identity()["cache_key"], changed_lock)
                instance.manifest["files"][0]["sha256"] = "f" * 64
                changed_code = instance.data_identity()["cache_key"]
                self.assertNotEqual(changed_code, changed_lock)
                (raw / names[2]).write_bytes(b"changed sound speed profile")
                self.assertNotEqual(instance.data_identity()["cache_key"], changed_code)


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="swellex-dispatch-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.local = self.root / RUN_ID
        self.config = valid_config()
        self.manifest = {"schema_version": 1, "run_id": RUN_ID, "suite": "quick", "files": [],
                         "source_sha256": "0" * 64, "config_sha256": "1" * 64,
                         "git_head": "fixture", "git_status": [" M README.md"]}

    def patches(self, remote_call, rsync_code=0):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(remote, "LOCAL_RUNS", self.root))
        stack.enter_context(mock.patch.object(remote, "freeze_source", return_value=self.manifest))
        clock = stack.enter_context(mock.patch.object(remote, "datetime"))
        clock.now.return_value.strftime.return_value = RUN_ID[:-8]
        clock.now.return_value.isoformat.return_value = "2026-10-08T12:00:00+00:00"
        identifier = stack.enter_context(mock.patch.object(remote.uuid, "uuid4"))
        identifier.return_value.hex = "01234567" + "8" * 24
        stack.enter_context(mock.patch.object(remote, "remote_python", side_effect=remote_call))
        process = subprocess.CompletedProcess([], rsync_code, stdout="", stderr="injected rsync failure" if rsync_code else "")
        run = stack.enter_context(mock.patch.object(remote.subprocess, "run", return_value=process))
        stack.enter_context(mock.patch("builtins.print"))
        return run

    def assert_evidence_precedes_network(self):
        for name in ("manifest.json", "request.json", "generation_config.json", "dispatch.json", "report.json"):
            self.assertTrue((self.local / name).is_file(), name)
        self.assertEqual(json.loads((self.local / "manifest.json").read_text()), self.manifest)
        self.assertEqual(json.loads((self.local / "report.json").read_text())["status"], "STARTING")

    def failed_report(self):
        report = json.loads((self.local / "report.json").read_text())
        self.assertEqual(report["run_id"], RUN_ID)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["exit_code"], 1)
        self.assertIn("finished_at", report)
        return report

    def test_unreachable_preflight_retains_manifest_request_and_failure(self):
        def call(cfg, code, payload):
            self.assertEqual(code, remote.CHECK_CODE)
            self.assert_evidence_precedes_network()
            raise RuntimeError("network unreachable")
        run = self.patches(call)
        with self.assertRaisesRegex(RuntimeError, "network unreachable"):
            remote.validate(self.config, "quick", True, False, False)
        run.assert_not_called()
        report = self.failed_report()
        self.assertNotIn("remote_recovery", report)
        self.assertIn("network unreachable", report["error"])

    def test_source_transfer_failure_marks_owned_remote_run_failed(self):
        calls = []
        tokens = []
        def call(cfg, code, payload):
            calls.append(code)
            if code == remote.CHECK_CODE:
                self.assert_evidence_precedes_network()
                return {"status": "PASS"}
            if code == remote.INIT_CODE:
                self.assertEqual(payload["report"]["status"], "STARTING")
                self.assertEqual(payload["manifest.json"], self.manifest)
                tokens.append(payload["dispatch_token"])
                return {"run_dir": remote.run_dir(cfg, RUN_ID)}
            self.assertEqual(code, remote.DISPATCH_FAILURE_CODE)
            self.assertEqual(payload["dispatch_token"], tokens[0])
            return {"marked": True, "report": copy.deepcopy(payload["report"])}
        run = self.patches(call, rsync_code=23)
        with self.assertRaisesRegex(RuntimeError, "Source transfer failed"):
            remote.validate(self.config, "quick", True, False, False)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(calls, [remote.CHECK_CODE, remote.INIT_CODE, remote.DISPATCH_FAILURE_CODE])
        self.assertTrue(self.failed_report()["remote_recovery"]["marked"])

    def test_lost_initialization_response_attempts_owned_run_recovery(self):
        def call(cfg, code, payload):
            if code == remote.CHECK_CODE:
                return {"status": "PASS"}
            if code == remote.INIT_CODE:
                raise RuntimeError("initialization response lost")
            self.assertEqual(code, remote.DISPATCH_FAILURE_CODE)
            return {"marked": True, "report": copy.deepcopy(payload["report"])}
        run = self.patches(call)
        with self.assertRaisesRegex(RuntimeError, "initialization response lost"):
            remote.validate(self.config, "quick", True, False, False)
        run.assert_not_called()
        self.assertTrue(self.failed_report()["remote_recovery"]["marked"])

    def test_ambiguous_launch_failure_preserves_active_worker(self):
        def call(cfg, code, payload):
            if code == remote.CHECK_CODE:
                return {"status": "PASS"}
            if code == remote.INIT_CODE:
                return {"run_dir": remote.run_dir(cfg, RUN_ID)}
            if code == remote.START_CODE:
                raise RuntimeError("launch response lost")
            self.assertEqual(code, remote.DISPATCH_FAILURE_CODE)
            return {"marked": False, "report": {"run_id": RUN_ID, "status": "RUNNING"}}
        self.patches(call)
        with self.assertRaisesRegex(RuntimeError, "launch response lost"):
            remote.validate(self.config, "quick", True, False, False)
        result = self.failed_report()["remote_recovery"]
        self.assertFalse(result["marked"])
        self.assertEqual(result["report"]["status"], "RUNNING")

    def test_recovery_network_failure_still_persists_local_terminal_report(self):
        def call(cfg, code, payload):
            if code == remote.CHECK_CODE:
                return {"status": "PASS"}
            if code == remote.INIT_CODE:
                return {"run_dir": remote.run_dir(cfg, RUN_ID)}
            raise RuntimeError("network unavailable during recovery")
        self.patches(call, rsync_code=23)
        with self.assertRaisesRegex(RuntimeError, "Source transfer failed"):
            remote.validate(self.config, "quick", True, False, False)
        self.assertIn("network unavailable during recovery", self.failed_report()["remote_recovery_error"])

    def test_existing_local_run_is_never_overwritten(self):
        self.local.mkdir()
        report = self.local / "report.json"
        report.write_text('{"status":"PASS","marker":"original"}')
        self.patches(lambda *args: self.fail("Network must not be contacted for an existing run"))
        with self.assertRaises(FileExistsError):
            remote.validate(self.config, "quick", True, False, False)
        self.assertEqual(json.loads(report.read_text()), {"status": "PASS", "marker": "original"})

    def test_failure_marker_cannot_overwrite_foreign_or_launched_run(self):
        run = self.root / "owned"
        run.mkdir()
        report = run / "report.json"
        starting = {"run_id": RUN_ID, "status": "STARTING"}
        report.write_text(json.dumps(starting))
        (run / "dispatch.json").write_text(json.dumps({"token": "owner"}))
        def recover(token):
            payload = {"run_dir": str(run), "dispatch_token": token,
                       "report": {"run_id": RUN_ID, "status": "FAIL", "exit_code": 1}}
            proc = subprocess.run([sys.executable, "-c", remote.DISPATCH_FAILURE_CODE], input=json.dumps(payload),
                                  text=True, capture_output=True, check=True)
            return json.loads(proc.stdout)
        self.assertFalse(recover("other-owner")["marked"])
        self.assertEqual(json.loads(report.read_text()), starting)
        (run / "launch.json").write_text("{}")
        self.assertFalse(recover("owner")["marked"])
        self.assertEqual(json.loads(report.read_text()), starting)
        (run / "launch.json").unlink()
        self.assertTrue(recover("owner")["marked"])
        self.assertEqual(json.loads(report.read_text())["status"], "FAIL")
        self.assertEqual(json.loads((run / "export/data/report.json").read_text())["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
