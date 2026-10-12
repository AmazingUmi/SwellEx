"""Selected-run and shell contracts; execute these tests on haiqin."""
from __future__ import annotations

import ast
import copy
import hashlib
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
PROJECT = WORKFLOW.parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.localization_networks.workflow import config, remote, tasks

RUN_ID = "20261008T120000Z-01234567"


def valid_config():
    return copy.deepcopy(config.load_config(WORKFLOW / "haiqin_full_interleaved.toml"))


class PlanTests(unittest.TestCase):
    def test_all_fifteen_mappings_match_current_registries(self):
        selections = []
        for method in tasks.METHODS:
            tree = ast.parse((PROJECT / "scripts/localization_networks" / f"{method.lower()}/network/model.py").read_text())
            registry = next(node.value for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "MODEL_REGISTRY" for target in node.targets))
            registered = {ast.literal_eval(key) for key in registry.keys}
            for architecture in tasks.ARCHITECTURES[:-1]:
                with self.subTest(method=method, architecture=architecture):
                    item = tasks.resolve_selection(method, architecture)
                    self.assertEqual(set(item), {"method", "architecture", "model"})
                    self.assertIn(item["model"], registered)
                    self.assertEqual(tasks.validate_selection(item), item)
                    self.assertEqual(tasks.model_plan("selected", item), [item])
                    selections.append(item)
        for method in tasks.METHODS:
            grnn = tasks.resolve_selection(method, "GRNN")
            self.assertEqual(grnn["model"], f"{method.lower()}_grnn_range")
            self.assertEqual(tasks.entry_script(grnn), "scripts/localization_networks/grnn/main.py")
            self.assertEqual(tasks.validate_selection(grnn), grnn)
            selections.append(grnn)
        self.assertEqual(len({item["model"] for item in selections}), 15)

    def test_noncanonical_and_malformed_selections_rejected(self):
        for method, architecture in (("rbd", "complex_CNN"), ("RBD", "complex_cnn"), ("RBD", "ResNet34"), (None, "ResNet18"), ("SCM", [])):
            with self.subTest(method=method, architecture=architecture), self.assertRaises(ValueError):
                tasks.resolve_selection(method, architecture)
        baseline = tasks.resolve_selection("ELM", "ResNet18")
        candidates = [None, [], {**baseline, "extra": True}, {k: v for k, v in baseline.items() if k != "model"}, {**baseline, "model": "resnet18_range"}, {**baseline, "architecture": "ResNet50"}]
        for item in candidates:
            with self.subTest(selection=item), self.assertRaises(ValueError):
                tasks.validate_selection(item)

    def test_suite_plans_preserve_validation_scope(self):
        self.assertEqual(tasks.model_plan("quick"), [])
        self.assertEqual(tasks.model_plan("smoke"), [tasks.resolve_selection(method, "complex_CNN") for method in tasks.METHODS] + [tasks.resolve_selection(method, "GRNN") for method in tasks.METHODS])
        experiment = tasks.model_plan("experiment")
        self.assertEqual(len(experiment), 15)
        self.assertEqual(experiment[-1], tasks.resolve_selection("SCM", "GRNN"))
        self.assertEqual(len({item["model"] for item in experiment}), 15)
        for suite, selection in (("unknown", None), ("selected", None), ("smoke", experiment[0]), ("quick", experiment[0]), ("experiment", experiment[0])):
            with self.subTest(suite=suite), self.assertRaises(ValueError):
                tasks.model_plan(suite, selection)

    def test_phase_sets_distinguish_validation_from_selected(self):
        base = {"snapshot_integrity", "environment", "source_unchanged_after_validation"}
        quick = base | {"python_quick", "data_numerical"}
        self.assertEqual(tasks.expected_phases("quick"), quick)
        data = {"original_data_identity", "data_generation", "dataset_contract", "visualization"}
        for suite in ("smoke", "experiment"):
            evaluations = {f"evaluate_{item['method']}_{item['model']}" for item in tasks.model_plan(suite)}
            self.assertEqual(tasks.expected_phases(suite), quick | data | evaluations | {"checkpoint_resume"})
        for method, architecture, evaluation in (("ELM", "ResNet50", "evaluate_ELM_elm_resnet50_range"), ("RBD", "GRNN", "evaluate_RBD_rbd_grnn_range"), ("ELM", "GRNN", "evaluate_ELM_elm_grnn_range"), ("SCM", "GRNN", "evaluate_SCM_scm_grnn_range")):
            self.assertEqual(tasks.expected_phases("selected", tasks.resolve_selection(method, architecture)), base | data | {evaluation})


class SelectedCliTests(unittest.TestCase):
    def test_valid_overrides_freeze_config_without_mutating_original(self):
        original = valid_config()
        before = copy.deepcopy(original)
        with mock.patch.object(remote, "dispatch", return_value=0) as dispatch:
            self.assertEqual(remote.selected_run(original, "ELM", "ResNet50", True, False, True, epochs=7, seed=0), 0)
        frozen, suite, detach, dry_run, no_cache = dispatch.call_args.args
        self.assertEqual((suite, detach, dry_run, no_cache), ("selected", True, False, True))
        self.assertEqual(frozen["training"]["epochs"], 7)
        self.assertEqual(frozen["validation"]["seed"], 0)
        self.assertEqual(dispatch.call_args.kwargs["selection"], tasks.resolve_selection("ELM", "ResNet50"))
        self.assertEqual(original, before)

    def test_grnn_epochs_and_invalid_overrides_reject_before_dispatch(self):
        cases = [(method, "GRNN", {"epochs": 1}) for method in tasks.METHODS] + [("RBD", "real_CNN", {"epochs": 0}), ("RBD", "real_CNN", {"epochs": True}), ("RBD", "real_CNN", {"seed": -1}), ("RBD", "real_CNN", {"seed": False}), ("RBD", "real_CNN", {"seed": 2**32}), ("SCM", "GRNN", {"seed": 2**32})]
        with mock.patch.object(remote, "dispatch") as dispatch, mock.patch.object(remote, "remote_python") as ssh:
            for method, architecture, options in cases:
                with self.subTest(method=method, architecture=architecture, options=options), self.assertRaises(ValueError):
                    remote.selected_run(valid_config(), method, architecture, False, False, False, **options)
        dispatch.assert_not_called()
        ssh.assert_not_called()

    def test_each_grnn_method_routes_to_its_canonical_selection(self):
        for method in tasks.METHODS:
            with self.subTest(method=method), mock.patch.object(remote, "dispatch", return_value=0) as dispatch:
                self.assertEqual(remote.selected_run(valid_config(), method, "GRNN", False, True, False), 0)
                self.assertEqual(dispatch.call_args.kwargs["selection"], tasks.resolve_selection(method, "GRNN"))

    def test_config_placement_for_all_commands(self):
        parser = remote.make_parser()
        path = Path("/tmp/experiment config.toml")
        commands = [("run", "SCM", "GRNN"), ("validate",), ("check",), ("status", "--run", RUN_ID), ("fetch", "--run", RUN_ID)]
        for command in commands:
            for argv in (["--config", str(path), *command], [*command, "--config", str(path)]):
                with self.subTest(argv=argv):
                    self.assertEqual(parser.parse_args(argv).config, path)

    def test_unknown_and_obsolete_options_reject_before_ssh(self):
        invalid = [["run", "rbd", "real_CNN"], ["run", "RBD", "complex_cnn_range"], ["run", "SCM", "GRNN", "--lr", "0.1"], ["run", "RBD", "real_CNN", "--epoch", "5"], ["run", "SCM", "GRNN", "--epochs", "bad"], ["validate", "--suite", "selected"]]
        with mock.patch.object(remote, "remote_python") as ssh, mock.patch.object(sys, "stderr", io.StringIO()):
            for argv in invalid:
                with self.subTest(argv=argv), self.assertRaises(SystemExit) as error:
                    remote.main(argv)
                self.assertEqual(error.exception.code, 2)
        ssh.assert_not_called()

    def test_main_grnn_epochs_rejects_without_ssh_or_run_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(remote, "LOCAL_RUNS", Path(directory) / "data"), mock.patch.object(remote, "remote_python") as ssh, mock.patch.object(sys, "stderr", io.StringIO()):
            self.assertEqual(remote.main(["run", "SCM", "GRNN", "--epochs", "1"]), 1)
            self.assertFalse((Path(directory) / "data").exists())
        ssh.assert_not_called()

    def test_validation_default_remains_smoke_and_unmodified_config(self):
        parser = remote.make_parser()
        self.assertEqual(parser.parse_args(["validate"]).suite, "smoke")
        cfg = valid_config()
        self.assertEqual(cfg["training"]["epochs"], 100)
        self.assertEqual((cfg["dataset"]["segment_start_s"], cfg["dataset"]["segment_end_s"]), (0, 4499))
        with mock.patch.object(remote, "dispatch", return_value=0) as dispatch:
            self.assertEqual(remote.validate(cfg, "smoke", False, True, False), 0)
        self.assertEqual(dispatch.call_args.args, (cfg, "smoke", False, True, False))
        self.assertEqual(dispatch.call_args.kwargs, {})


class FrozenSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swellex-selected-freeze-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "source"
        self.project.mkdir()
        subprocess.run(["git", "init", "-q", str(self.project)], check=True, capture_output=True)
        worker = self.project / "scripts/localization_networks/workflow/worker.py"
        worker.parent.mkdir(parents=True)
        worker.write_text("# fixture worker\n")
        subprocess.run(["git", "add", "."], cwd=self.project, check=True, capture_output=True)
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], cwd=self.project, check=True, capture_output=True)

    def test_selected_manifest_contains_canonical_selection_and_config_hash(self):
        cfg = valid_config()
        cfg["training"]["epochs"] = 9
        cfg["validation"]["seed"] = 42
        selection = tasks.resolve_selection("SCM", "GRNN")
        manifest = remote.freeze_source(self.root / "selected", cfg, RUN_ID, "selected", self.project, selection=selection)
        self.assertEqual(manifest["selection"], selection)
        self.assertEqual(manifest["config_sha256"], hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest())
        old = remote.freeze_source(self.root / "quick", cfg, RUN_ID, "quick", self.project)
        self.assertNotIn("selection", old)

    def test_snapshot_preserves_executable_shell_entry(self):
        source = self.project / "scripts/run.sh"
        source.write_text("#!/bin/sh\nexit 0\n")
        source.chmod(0o755)
        stage = self.root / "executable"
        manifest = remote.freeze_source(stage, valid_config(), RUN_ID, "quick", self.project)
        self.assertIn("scripts/run.sh", {item["path"] for item in manifest["files"]})
        self.assertEqual((stage / "scripts/run.sh").stat().st_mode & 0o777, 0o755)
        subprocess.run([str(stage / "scripts/run.sh")], check=True, capture_output=True)

    def test_selected_request_and_dry_run_have_plan_overrides_and_no_network(self):
        cfg = valid_config()
        selection = tasks.resolve_selection("RBD", "ResNet18")
        output = io.StringIO()
        original_freeze = remote.freeze_source
        def freeze(stage, frozen, run_id, suite, *, selection=None):
            return original_freeze(stage, frozen, run_id, suite, self.project, selection=selection)
        with mock.patch.object(remote, "freeze_source", side_effect=freeze), mock.patch.object(remote, "LOCAL_RUNS", self.root / "data"), mock.patch.object(remote, "LOCAL_FIGURES", self.root / "figure"), mock.patch.object(remote, "remote_python") as ssh, mock.patch.object(sys, "stdout", output):
            self.assertEqual(remote.selected_run(cfg, "RBD", "ResNet18", False, True, True, epochs=11, seed=13), 0)
        ssh.assert_not_called()
        self.assertFalse((self.root / "data").exists())
        self.assertFalse((self.root / "figure").exists())
        result = json.loads(output.getvalue())
        request = result["request"]
        self.assertEqual(set(request), {"schema_version", "run_id", "suite", "config", "use_cache", "selection"})
        self.assertEqual(request["selection"], selection)
        self.assertEqual(result["model_plan"], [selection])
        self.assertEqual(set(result["expected_phases"]), tasks.expected_phases("selected", selection))
        self.assertEqual(request["config"]["training"]["epochs"], 11)
        self.assertEqual(request["config"]["validation"]["seed"], 13)
        self.assertFalse(request["use_cache"])
        self.assertEqual(result["generation_config"]["dataset"], cfg["dataset"])

    def test_dispatch_freezes_merged_request_and_manifest_before_network(self):
        cfg = valid_config()
        frozen = copy.deepcopy(cfg)
        frozen["training"]["epochs"] = 6
        frozen["validation"]["seed"] = 17
        selection = tasks.resolve_selection("SCM", "ResNet18")
        manifest = remote.freeze_source(self.root / "dispatch-stage", frozen, RUN_ID, "selected", self.project, selection=selection)
        received = []
        def network(frozen_config, code, payload):
            directories = list((self.root / "data").iterdir())
            self.assertEqual(len(directories), 1)
            local = directories[0]
            request = json.loads((local / "request.json").read_text())
            self.assertEqual(request["run_id"], manifest["run_id"])
            self.assertEqual(request["selection"], selection)
            self.assertEqual(request["config"], frozen)
            self.assertEqual(json.loads((local / "manifest.json").read_text())["selection"], selection)
            received.append(code)
            if code == remote.CHECK_CODE:
                return {"status": "PASS"}
            if code == remote.INIT_CODE:
                self.assertEqual(payload["request.json"], request)
                self.assertEqual(payload["manifest.json"], manifest)
                return {"run_dir": remote.run_dir(frozen_config, request["run_id"])}
            self.assertEqual(code, remote.START_CODE)
            return {"pid": 123}
        completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with mock.patch.object(remote, "LOCAL_RUNS", self.root / "data"), mock.patch.object(remote, "freeze_source", return_value=manifest) as freeze, mock.patch.object(remote, "remote_python", side_effect=network), mock.patch.object(remote.subprocess, "run", return_value=completed) as transfer, mock.patch.object(remote, "datetime") as clock, mock.patch.object(remote.uuid, "uuid4") as identifier, mock.patch("builtins.print"):
            clock.now.return_value.strftime.return_value = RUN_ID[:-8]
            clock.now.return_value.isoformat.return_value = "2026-10-08T12:00:00+00:00"
            identifier.return_value.hex = "01234567" + "a" * 24
            self.assertEqual(remote.selected_run(cfg, "SCM", "ResNet18", True, False, False, epochs=6, seed=17), 0)
        self.assertEqual(freeze.call_args.kwargs["selection"], selection)
        self.assertEqual(received, [remote.CHECK_CODE, remote.INIT_CODE, remote.START_CODE])
        self.assertEqual(transfer.call_args.args[0][1], "-rltp")


class ShellEntryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swellex shell fixture ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "scripts").mkdir()
        shutil.copyfile(PROJECT / "scripts/run.sh", self.root / "scripts/run.sh")
        self.capture = self.root / "capture.json"
        binary = self.root / "bin"
        binary.mkdir()
        uv = binary / "uv"
        uv.write_text(f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\nPath(os.environ['SWELLEX_CAPTURE']).write_text(json.dumps({{'cwd': os.getcwd(), 'argv': sys.argv[1:]}}))\nsys.exit(int(os.environ['SWELLEX_EXIT']))\n")
        uv.chmod(0o755)
        self.environment = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"], "SWELLEX_CAPTURE": str(self.capture), "SWELLEX_EXIT": "0"}

    def invoke(self, arguments, exit_code=0):
        self.environment["SWELLEX_EXIT"] = str(exit_code)
        result = subprocess.run(["sh", str(self.root / "scripts/run.sh"), *arguments], cwd=self.root.parent, env=self.environment, capture_output=True, text=True)
        captured = json.loads(self.capture.read_text())
        self.assertEqual(captured["cwd"], str(self.root))
        self.assertEqual(captured["argv"][:5], ["run", "--no-project", "--python", "python3", "scripts/localization_networks/workflow/remote.py"])
        return result, captured["argv"][5:]

    def test_selected_forwarding_preserves_spaced_and_shell_metacharacter_arguments(self):
        marker = self.root / "injection-marker"
        arguments = ["ELM", "ResNet50", "--config", f"config with spaces; touch {marker}.toml", "--seed", "0"]
        result, forwarded = self.invoke(arguments)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(forwarded, ["run", *arguments])
        self.assertFalse(marker.exists())

    def test_global_config_selection_and_management_commands(self):
        for args, expected in [(["--config", "experiment config.toml", "SCM", "GRNN"], ["--config", "experiment config.toml", "run", "SCM", "GRNN"]), (["status", "--run", RUN_ID, "--config", "a b.toml"], ["status", "--run", RUN_ID, "--config", "a b.toml"]), (["validate", "--suite", "quick"], ["validate", "--suite", "quick"]), (["--help"], ["--help"])]:
            with self.subTest(args=args):
                result, forwarded = self.invoke(args)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(forwarded, expected)

    def test_uv_exit_status_propagates(self):
        result, forwarded = self.invoke(["fetch", "--run", RUN_ID], exit_code=37)
        self.assertEqual(result.returncode, 37)
        self.assertEqual(forwarded, ["fetch", "--run", RUN_ID])


if __name__ == "__main__":
    unittest.main()
