"""Stdlib-only local GRNN routing; no numerical computation or SSH."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

WORKFLOW = Path(__file__).resolve().parents[1]
PROJECT = WORKFLOW.parents[3]
sys.path.insert(0, str(PROJECT))
from scripts.base_pack.localization_networks.workflow import remote


class LocalGrnnRoutingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swellex local grnn routing ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def config(self, side="before"):
        return WORKFLOW / f"haiqin_range_overlap_train_{side}.toml"

    def test_dry_run_all_methods_and_both_directions_preserve_spaced_paths(self):
        data = self.root / "method data; literal $(token)"
        python = self.root / "CPU env/bin/python"
        for method, side in ((method, side) for method in remote.METHODS for side in ("before", "after")):
            stdout = io.StringIO()
            with self.subTest(method=method, side=side), mock.patch.object(sys, "stdout", stdout), \
                 mock.patch.object(remote, "remote_python") as ssh, \
                 mock.patch.object(remote, "freeze_source") as freeze, \
                 mock.patch.object(remote.subprocess, "run") as launch, \
                 mock.patch.object(remote, "LOCAL_RUNS", self.root / "data"), \
                 mock.patch.object(remote, "LOCAL_FIGURES", self.root / "figures"):
                code = remote.main(["local-grnn", "--method", method, "--config", str(self.config(side)),
                                    "--python", str(python), "--data", str(data), "--dry-run"])
            self.assertEqual(code, 0)
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan["execution"], "local_cpu")
            self.assertEqual(plan["config"]["dataset"]["train_side"], side)
            self.assertEqual(plan["selection"], remote.resolve_selection(method, "GRNN"))
            self.assertEqual(plan["argv"], [str(python), "-B", str(WORKFLOW / "local_grnn.py"),
                                           "--method", method, "--config", str(self.config(side)), "--data", str(data.resolve())])
            self.assertEqual(plan["data"], "reuse")
            ssh.assert_not_called()
            freeze.assert_not_called()
            launch.assert_not_called()
            self.assertFalse((self.root / "data").exists())
            self.assertFalse((self.root / "figures").exists())

    def test_config_placement_and_method_and_python_are_required(self):
        parser = remote.make_parser()
        for argv in (["--config", str(self.config()), "local-grnn", "--method", "SCM", "--python", sys.executable],
                     ["local-grnn", "--method", "SCM", "--python", sys.executable, "--config", str(self.config())]):
            with self.subTest(argv=argv):
                self.assertEqual(parser.parse_args(argv).config, self.config())
        for options in (["--method", "SCM"], ["--python", sys.executable],
                        ["--method", "scm", "--python", sys.executable]):
            with self.subTest(options=options), mock.patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                parser.parse_args(["local-grnn", "--config", str(self.config()), "--dry-run", *options])
            self.assertEqual(error.exception.code, 2)

    def test_invalid_split_and_unknown_config_reject_before_any_launch(self):
        unknown = self.root / "unknown.toml"
        unknown.write_text(self.config().read_text().replace("schema_version = 1", "schema_version = 1\nunknown = 1"))
        for path in (WORKFLOW / "haiqin_full_interleaved.toml", unknown):
            with self.subTest(path=path), mock.patch.object(sys, "stderr", io.StringIO()), \
                 mock.patch.object(remote, "remote_python") as ssh, \
                 mock.patch.object(remote.subprocess, "run") as launch:
                self.assertEqual(remote.main(["local-grnn", "--method", "SCM", "--config", str(path),
                                              "--python", sys.executable, "--dry-run"]), 1)
            ssh.assert_not_called()
            launch.assert_not_called()

    def test_remote_training_options_are_rejected_for_local_route(self):
        for options in (("--epochs", "1"), ("--detach",), ("--no-cache",)):
            with self.subTest(options=options), mock.patch.object(sys, "stderr", io.StringIO()), \
                 mock.patch.object(remote.subprocess, "run") as launch, \
                 self.assertRaises(SystemExit) as error:
                remote.main(["local-grnn", "--method", "SCM", "--config", str(self.config()),
                             "--python", sys.executable, *options])
            self.assertEqual(error.exception.code, 2)
            launch.assert_not_called()

    def test_execute_uses_selected_venv_path_and_propagates_exit_status(self):
        python = self.root / "CPU env/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        completed = subprocess.CompletedProcess([], 37)
        with mock.patch.object(remote.subprocess, "run", return_value=completed) as launch, \
             mock.patch.object(remote, "remote_python") as ssh:
            code = remote.main(["local-grnn", "--method", "RBD", "--config", str(self.config()), "--python", str(python)])
        self.assertEqual(code, 37)
        launch.assert_called_once_with(
            [str(python), "-B", str(WORKFLOW / "local_grnn.py"), "--method", "RBD", "--config", str(self.config())],
            cwd=PROJECT, check=False)
        ssh.assert_not_called()

    def test_import_and_dry_run_do_not_import_numerical_worker(self):
        # An isolated interpreter makes this check independent of test ordering.
        code = """import importlib.abc, sys
class RejectNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'scipy', 'h5py', 'matplotlib'}:
            raise AssertionError('Numerical import in dispatcher: ' + fullname)
sys.meta_path.insert(0, RejectNumericalImports())
from scripts.base_pack.localization_networks.workflow import remote
result = remote.main(sys.argv[1:])
assert 'scripts.base_pack.localization_networks.workflow.local_grnn' not in sys.modules
assert 'scripts.base_pack.data_processing.generation_core' not in sys.modules
raise SystemExit(result)
"""
        process = subprocess.run([sys.executable, "-B", "-c", code, "local-grnn", "--method", "ELM", "--config", str(self.config()),
                                  "--python", sys.executable, "--dry-run"], cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["data"], "generate_elm")


class LocalGrnnShellEntryTests(unittest.TestCase):
    def test_shell_preserves_local_command_and_all_literal_arguments(self):
        with tempfile.TemporaryDirectory(prefix="swellex local shell ") as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            shutil.copyfile(PROJECT / "scripts/run.sh", root / "scripts/run.sh")
            (root / "bin").mkdir()
            capture = root / "capture.json"
            uv = root / "bin/uv"
            uv.write_text(f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
                          "Path(os.environ['SWELLEX_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n")
            uv.chmod(0o755)
            environment = {**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                           "SWELLEX_CAPTURE": str(capture)}
            argv = ["base_pack", "local-grnn", "--method", "RBD", "--config", "config with spaces.toml", "--python", "CPU env/bin/python",
                    "--data", "RBD data; $(literal)", "--dry-run"]
            result = subprocess.run(["sh", str(root / "scripts/run.sh"), *argv],
                                    cwd=root.parent, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(capture.read_text()),
                             ["run", "--no-project", "--python", "python3", "python", "-B",
                              "scripts/run.py", *argv])


if __name__ == "__main__":
    unittest.main()
