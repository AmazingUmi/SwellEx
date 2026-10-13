"""Pack routing and isolation contracts; standard-library only."""
from __future__ import annotations

import io
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.packs import PACKS, get_pack
from scripts.run import PROJECT, main, pack_arguments
from scripts.base_pack.localization_networks.workflow import remote
from scripts.base_pack.localization_networks.workflow.compare_results import run_records
from scripts.base_pack.visualization.results import load_run


class PackRoutingTests(unittest.TestCase):
    def test_pack_only_runs_full_configured_experiment(self):
        for pack, expected in (("base_pack", 15), ("joint_feature_pack", 35)):
            dispatcher = import_module(get_pack(pack).dispatcher)
            with self.subTest(pack=pack), patch.object(dispatcher, "dispatch", return_value=0) as dispatch:
                self.assertEqual(main([pack]), 0)
            config, suite, detach, dry_run, no_cache = dispatch.call_args.args
            self.assertEqual((suite, detach, dry_run, no_cache), ("experiment", False, False, False))
            self.assertEqual(len(dispatcher.model_plan(suite)), expected)
            self.assertEqual(config["training"]["epochs"], 100)
            self.assertEqual(config["dataset"]["segment_end_s"], 4499)
            if pack == "joint_feature_pack":
                self.assertEqual(config["fusion"]["mode"], "block_rms")

    def test_default_experiment_options_and_config_placement(self):
        for pack in PACKS.values():
            dispatcher = import_module(pack.dispatcher)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config with spaces; $(literal).toml"
                path.write_text(dispatcher.DEFAULT_CONFIG.read_text().replace("epochs = 100", "epochs = 3"))
                variants = (["--config", str(path), "--dry-run", "--detach", "--no-cache"],
                            ["--dry-run", "--config", str(path), "--detach", "--no-cache"],
                            [f"--config={path}", "--dry-run", "--detach", "--no-cache"])
                for arguments in variants:
                    with self.subTest(pack=pack.name, arguments=arguments), patch.object(dispatcher, "dispatch", return_value=0) as dispatch:
                        self.assertEqual(main([pack.name, *arguments]), 0)
                    config, *settings = dispatch.call_args.args
                    self.assertEqual(settings, ["experiment", True, True, True])
                    self.assertEqual(config["training"]["epochs"], 3)

    def test_explicit_suite_and_selected_model_do_not_expand_to_full_plan(self):
        for pack in PACKS.values():
            dispatcher = import_module(pack.dispatcher)
            cases = [(["validate"], "smoke"), (["--suite", "quick"], "quick"),
                     (["validate", "--suite=experiment"], "experiment"),
                     ([pack.methods[-1], "ResNet18", "--epochs", "2"], "selected")]
            for arguments, expected in cases:
                with self.subTest(pack=pack.name, arguments=arguments), patch.object(dispatcher, "dispatch", return_value=0) as dispatch:
                    self.assertEqual(main([pack.name, *arguments]), 0)
                self.assertEqual(dispatch.call_args.args[1], expected)
                if expected == "selected":
                    selection = dispatch.call_args.kwargs["selection"]
                    self.assertEqual(selection["method"], pack.methods[-1])
                    self.assertEqual(selection["architecture"], "ResNet18")
                    self.assertEqual(dispatch.call_args.args[0]["training"]["epochs"], 2)

    def test_joint_fusion_override_and_base_rejection(self):
        dispatcher = import_module(get_pack("joint_feature_pack").dispatcher)
        with patch.object(dispatcher, "dispatch", return_value=0) as dispatch:
            self.assertEqual(main(["joint_feature_pack", "--fusion", "raw_concat", "--dry-run"]), 0)
        self.assertEqual(dispatch.call_args.args[0]["fusion"]["mode"], "raw_concat")
        with patch.object(remote, "dispatch") as dispatch, patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["base_pack", "--fusion", "raw_concat"])
        self.assertEqual(raised.exception.code, 2)
        dispatch.assert_not_called()

    def test_invalid_or_duplicate_arguments_never_launch(self):
        invalid = (["RBD"], ["RBD", "invalid"], ["typo"], ["--unknown"],
                   ["--config"], ["--config", "--dry-run"], ["--epochs", "2"],
                   ["--config=a", "validate", "--config", "b"],
                   ["--dry-run", "--dry-run"], ["--suite", "quick", "--suite=experiment"],
                   ["RBD", "ResNet18", "--seed", "0", "--seed=1"],
                   ["status"], ["fetch", "--run", "a", "--run", "b"])
        for pack in PACKS.values():
            dispatcher = import_module(pack.dispatcher)
            for arguments in invalid:
                with self.subTest(pack=pack.name, arguments=arguments), patch.object(dispatcher, "dispatch") as dispatch, patch.object(dispatcher, "remote_python") as ssh, patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        main([pack.name, *arguments])
                    self.assertEqual(raised.exception.code, 2)
                dispatch.assert_not_called()
                ssh.assert_not_called()

    def test_help_is_side_effect_free_and_describes_shell_defaults(self):
        for pack in PACKS.values():
            dispatcher = import_module(pack.dispatcher)
            for arguments in (["--help"], ["--config", "missing.toml", "--help"],
                              ["validate", "--help"], [pack.methods[-1], "--help"]):
                output = io.StringIO()
                with self.subTest(pack=pack.name, arguments=arguments), patch("sys.stdout", output), patch.object(dispatcher, "load_config") as config, patch.object(dispatcher, "dispatch") as dispatch:
                    with self.assertRaises(SystemExit) as raised:
                        main([pack.name, *arguments])
                    self.assertEqual(raised.exception.code, 0)
                    self.assertIn(f"./scripts/run.sh {pack.name}", output.getvalue())
                config.assert_not_called()
                dispatch.assert_not_called()

    def test_administration_uses_existing_run_without_launch(self):
        for pack in PACKS.values():
            dispatcher = import_module(pack.dispatcher)
            for command in ("status", "fetch"):
                with self.subTest(pack=pack.name, command=command), patch.object(dispatcher, command, return_value={"run_id": "existing", "status": "PASS", "checks": []}) as action, patch.object(dispatcher, "dispatch") as dispatch, patch("sys.stdout", io.StringIO()):
                    self.assertEqual(main([pack.name, command, "--run", "existing"]), 0)
                self.assertEqual(action.call_args.args[1], "existing")
                dispatch.assert_not_called()

    def test_pack_is_required_and_names_are_canonical(self):
        for arguments in ([], ["RBD", "complex_CNN"], ["base"], ["BASE_PACK"]):
            with self.subTest(arguments=arguments), patch("sys.stderr", io.StringIO()), patch("scripts.run.import_module") as load:
                with self.assertRaises(SystemExit) as raised:
                    main(arguments)
                self.assertEqual(raised.exception.code, 2)
                load.assert_not_called()

    def test_all_registered_packs_route_to_their_own_dispatcher(self):
        for pack in PACKS.values():
            for command in ([pack.methods[-1], "complex_CNN", "--dry-run"],
                            ["status", "--run", "id"], ["fetch", "--run", "id"],
                            ["validate", "--suite", "quick"]):
                dispatcher = Mock()
                dispatcher.main.return_value = 37
                with self.subTest(pack=pack.name, command=command), patch("scripts.run.import_module", return_value=dispatcher) as load:
                    self.assertEqual(main([pack.name, *command]), 37)
                load.assert_called_once_with(pack.dispatcher)
                dispatcher.main.assert_called_once_with(pack_arguments(pack, command))

    def test_spaced_paths_and_shell_syntax_are_preserved(self):
        pack = get_pack("joint_feature_pack")
        path = "folder with spaces; $(literal)/config.toml"
        self.assertEqual(pack_arguments(pack, ["--config", path, "RBD_ELM_SCM", "ResNet18"]),
                         ["--config", path, "run", "RBD_ELM_SCM", "ResNet18"])
        self.assertEqual(pack_arguments(pack, [f"--config={path}", "RBD_ELM_SCM", "ResNet18"]),
                         [f"--config={path}", "run", "RBD_ELM_SCM", "ResNet18"])

    def test_base_pack_does_not_accept_joint_method_shorthand(self):
        self.assertEqual(pack_arguments(get_pack("base_pack"), ["RBD_ELM", "real_CNN"]),
                         ["RBD_ELM", "real_CNN"])

    def test_imports_do_not_load_numerical_dependencies(self):
        code = """import importlib.abc, sys
class RejectNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'scipy', 'h5py', 'matplotlib'}:
            raise AssertionError('Numerical dispatcher import: ' + fullname)
sys.meta_path.insert(0, RejectNumericalImports())
from scripts.run import main
main(['base_pack', '--help'])
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class PackResultIsolationTests(unittest.TestCase):
    def test_base_analysis_does_not_mix_another_packs_results(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "report.json").write_text(json.dumps({
                "schema_version": 1, "pack": "joint_feature_pack", "status": "PASS",
            }))
            self.assertEqual(run_records(run, {}), [])
            with self.assertRaisesRegex(ValueError, "another experiment pack"):
                load_run(run)

    def test_status_rejects_wrong_or_missing_pack_and_run_identity(self):
        run_id = "20261013T120000Z-01234567"
        config = remote.load_config(remote.DEFAULT_CONFIG)
        correct = {"pack": "base_pack", "run_id": run_id, "status": "PASS"}
        for candidate in ({**correct, "pack": "joint_feature_pack"},
                          {key: value for key, value in correct.items() if key != "pack"},
                          {**correct, "run_id": "20261013T120000Z-aaaaaaaa"}):
            with self.subTest(candidate=candidate), patch.object(remote, "remote_python", return_value=candidate):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    remote.status(config, run_id)
        with patch.object(remote, "remote_python", return_value=correct):
            self.assertEqual(remote.status(config, run_id), correct)


if __name__ == "__main__":
    unittest.main()
