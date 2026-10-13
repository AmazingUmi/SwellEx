"""Selected-run worker contracts; execute on haiqin with mocked numerical work."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

WORKFLOW = Path(__file__).resolve().parents[1]
PROJECT = WORKFLOW.parents[3]
sys.path.insert(0, str(PROJECT))
from scripts.base_pack.localization_networks.workflow import config, tasks, worker

RUN_ID = "20261008T120000Z-01234567"


class SelectedWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="swellex-selected-worker-")
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve() / "workspace"
        self.run = self.workspace / "runs" / RUN_ID
        self.project = self.run / "project"
        self.project.mkdir(parents=True)
        self.config = copy.deepcopy(config.load_config(WORKFLOW / "haiqin_full_interleaved.toml"))
        self.config["remote"].update(workspace=str(self.workspace), repository=str(Path(self.temporary.name) / "source"))
        lock = self.project / "uv.lock"
        lock.write_text("frozen dependency fixture\n")
        self.lock_hash = worker.file_hash(lock)
        files = [{"path": "uv.lock", "size": lock.stat().st_size, "sha256": self.lock_hash}]
        self.manifest = {
            "pack": "base_pack", "run_id": RUN_ID, "suite": "selected", "git_head": "fixture", "files": files,
            "source_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "config_sha256": hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest(),
        }
        self.request = {"schema_version": 1, "pack": "base_pack", "run_id": RUN_ID, "suite": "selected", "config": self.config, "use_cache": True}
        self.generation = {"schema_version": 1, "dataset": self.config["dataset"], "rbd": self.config["rbd"], "result_path": "outputs/validation/generation_result.json"}
        self.select("ELM", "ResNet18")

    def select(self, method=None, architecture=None, *, suite="selected"):
        self.request["suite"] = self.manifest["suite"] = suite
        self.request.pop("selection", None)
        self.manifest.pop("selection", None)
        if suite == "selected":
            selection = tasks.resolve_selection(method, architecture)
            self.request["selection"] = selection
            self.manifest["selection"] = dict(selection)
        self.write_inputs()

    def write_inputs(self):
        for name, value in (("request.json", self.request), ("manifest.json", self.manifest), ("generation_config.json", self.generation)):
            worker.write_json(self.run / name, value)

    def prefix(self):
        return mock.patch.object(sys, "prefix", str(worker.workspace_environment(self.config, self.lock_hash)))

    def instance(self):
        return worker.Worker(self.run)

    def execute_mocked(self, instance, *, environment_failure=False):
        def datasets():
            instance.dataset_dirs = {method: str(self.project / "outputs/Datasets" / method) for method in ("RBD", "ELM", "SCM")}
            instance.phase("data_generation", lambda: {"status": "PASS"})
            instance.phase("dataset_contract", lambda: {"status": "PASS"})

        def metric(method, model):
            return {"method": method, "model": model, "source_segment_indices": [5, 10]}

        with ExitStack() as stack:
            stack.enter_context(self.prefix())
            environment = stack.enter_context(mock.patch.object(instance, "environment", side_effect=RuntimeError("GPU unavailable") if environment_failure else None, return_value={"cuda_available": True}))
            quick = stack.enter_context(mock.patch.object(instance, "json_command", return_value={"status": "PASS"}))
            identity = stack.enter_context(mock.patch.object(instance, "data_identity", return_value={"inputs": 24}))
            stack.enter_context(mock.patch.object(instance, "datasets", side_effect=datasets))
            neural = stack.enter_context(mock.patch.object(instance, "train_predict", side_effect=metric))
            grnn = stack.enter_context(mock.patch.object(instance, "grnn", side_effect=lambda method: metric(method, tasks.resolve_selection(method, "GRNN")["model"])))
            resume = stack.enter_context(mock.patch.object(instance, "checkpoint_resume"))
            visualization = stack.enter_context(mock.patch.object(instance, "visualize", return_value={"status": "PASS"}))
            code = instance.execute()
        return code, {"environment": environment, "quick": quick, "identity": identity, "neural": neural, "grnn": grnn, "resume": resume, "visualization": visualization}

    def test_selected_neural_execute_runs_exactly_requested_model(self):
        instance = self.instance()
        code, calls = self.execute_mocked(instance)
        self.assertEqual(code, 0)
        calls["neural"].assert_called_once_with("ELM", "elm_resnet18_range")
        calls["grnn"].assert_not_called()
        calls["resume"].assert_not_called()
        calls["quick"].assert_not_called()
        calls["identity"].assert_called_once()
        calls["visualization"].assert_called_once()
        self.assertEqual(instance.report["selection"], self.request["selection"])
        self.assertEqual(instance.report["model_plan"], [self.request["selection"]])
        self.assertEqual(instance.report["suite"], "selected")
        self.assertEqual(instance.report["purpose"], "configured_single_model_without_accuracy_acceptance_threshold")
        self.assertEqual(instance.report["not_run"], [])
        self.assertEqual(json.loads((self.run / "reports/metrics.json").read_text())["metrics"][0]["model"], "elm_resnet18_range")
        self.assertEqual(json.loads((self.run / "export/data/report.json").read_text())["status"], "PASS")

    def test_selected_grnn_execute_has_no_neural_training_or_resume(self):
        for method in tasks.METHODS:
            with self.subTest(method=method):
                self.select(method, "GRNN")
                instance = self.instance()
                code, calls = self.execute_mocked(instance)
                self.assertEqual(code, 0)
                calls["grnn"].assert_called_once_with(method)
                for name in ("neural", "resume", "quick"):
                    calls[name].assert_not_called()
                phases = {row["name"] for row in instance.report["checks"]}
                self.assertEqual({name for name in phases if name.startswith("evaluate_")},
                                 {f"evaluate_{method}_{method.lower()}_grnn_range"})
                self.assertEqual(instance.report["model_plan"], [self.request["selection"]])
                self.assertEqual(instance.report["not_run"], [])

    def test_existing_suites_preserve_validation_and_model_scope(self):
        for suite, neural_count, grnn_count, resume_count in (("quick", 0, 0, 0), ("smoke", 3, 3, 1), ("experiment", 12, 3, 1)):
            with self.subTest(suite=suite):
                self.select(suite=suite)
                instance = self.instance()
                code, calls = self.execute_mocked(instance)
                self.assertEqual(code, 0)
                self.assertEqual([call.args[0] for call in calls["quick"].call_args_list], ["python_quick", "data_numerical"])
                self.assertEqual(calls["neural"].call_count, neural_count)
                self.assertEqual(calls["grnn"].call_count, grnn_count)
                self.assertEqual(calls["grnn"].call_args_list,
                                 [] if suite == "quick" else [mock.call(method) for method in tasks.METHODS])
                self.assertEqual(calls["resume"].call_count, resume_count)
                if suite == "smoke":
                    self.assertEqual(calls["neural"].call_args_list, [mock.call("RBD", "complex_cnn_range"), mock.call("ELM", "elm_complex_cnn_range"), mock.call("SCM", "scm_complex_cnn_range")])
                if suite == "experiment":
                    self.assertIn(mock.call("SCM", "scm_resnet50_range"), calls["neural"].call_args_list)
                self.assertNotIn("selection", instance.report)
                self.assertEqual(instance.report["model_plan"], tasks.model_plan(suite))
                self.assertEqual(instance.report["not_run"], [])

    def test_selected_setup_rejects_schema_and_model_mapping_tampering(self):
        original = copy.deepcopy(self.request)
        mutations = (
            lambda r: r.pop("pack"),
            lambda r: r.update(pack="joint_feature_pack"),
            lambda r: r.pop("selection"),
            lambda r: r.update(unrecognized=True),
            lambda r: r.update(schema_version=True),
            lambda r: r["selection"].update(model="complex_cnn_range"),
            lambda r: r["selection"].update(method="RBD", architecture="GRNN", model="scm_grnn_range"),
            lambda r: r["selection"].update(architecture="resnet18"),
            lambda r: r["selection"].update(extra=True),
            lambda r: r["selection"].pop("model"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.request = copy.deepcopy(original)
                mutate(self.request)
                self.write_inputs()
                with self.prefix(), self.assertRaises(ValueError):
                    self.instance().setup()

    def test_manifest_cannot_substitute_another_experiment_pack(self):
        self.manifest["pack"] = "joint_feature_pack"
        self.write_inputs()
        with self.prefix(), self.assertRaisesRegex(ValueError, "Manifest pack"):
            self.instance().setup()

    def test_request_manifest_selection_must_be_identical_and_canonical(self):
        for value in (None, {"method": "ELM", "architecture": "ResNet18", "model": "elm_real_cnn_range"}, tasks.resolve_selection("SCM", "ResNet18")):
            with self.subTest(selection=value):
                if value is None:
                    self.manifest.pop("selection", None)
                else:
                    self.manifest["selection"] = value
                self.write_inputs()
                with self.prefix(), self.assertRaises(ValueError):
                    self.instance().setup()
        self.select(suite="quick")
        self.request["selection"] = tasks.resolve_selection("RBD", "complex_CNN")
        self.write_inputs()
        with self.prefix(), self.assertRaisesRegex(ValueError, "request schema"):
            self.instance().setup()
        self.request.pop("selection")
        self.manifest["selection"] = tasks.resolve_selection("RBD", "complex_CNN")
        self.write_inputs()
        with self.prefix(), self.assertRaisesRegex(ValueError, "manifests cannot"):
            self.instance().setup()

    def test_selected_failure_only_lists_selected_dependent_work(self):
        for method, architecture, evaluation in (("ELM", "ResNet18", "evaluate_ELM_elm_resnet18_range"), ("RBD", "GRNN", "evaluate_RBD_rbd_grnn_range"), ("ELM", "GRNN", "evaluate_ELM_elm_grnn_range"), ("SCM", "GRNN", "evaluate_SCM_scm_grnn_range")):
            with self.subTest(architecture=architecture):
                self.select(method, architecture)
                instance = self.instance()
                code, calls = self.execute_mocked(instance, environment_failure=True)
                self.assertEqual(code, 1)
                self.assertEqual(set(instance.report["not_run"]), {"original_data_identity", "data_generation", "dataset_contract", evaluation, "visualization", "source_unchanged_after_validation"})
                self.assertEqual(instance.report["checks"][-1]["status"], "FAIL")
                self.assertEqual(json.loads((self.run / "export/data/report.json").read_text()), instance.report)
                for name in ("quick", "identity", "neural", "grnn", "resume", "visualization"):
                    calls[name].assert_not_called()

    def test_failed_setup_keeps_conservative_snapshot_expectation(self):
        self.manifest["selection"] = tasks.resolve_selection("SCM", "GRNN")
        self.write_inputs()
        instance = self.instance()
        with self.prefix():
            self.assertEqual(instance.execute(), 1)
        self.assertEqual(instance.report["not_run"], [])
        self.assertEqual([(row["name"], row["status"]) for row in instance.report["checks"]], [("snapshot_integrity", "FAIL")])
        self.assertIn("selection differs", instance.report["error"])
        self.assertEqual(json.loads((self.run / "export/data/report.json").read_text()), instance.report)

    def prepare_operation(self):
        instance = self.instance()
        with self.prefix():
            instance.setup()
        instance.dataset_dirs = {method: str(self.project / "outputs/Datasets" / method) for method in ("RBD", "ELM", "SCM")}
        return instance

    def predictions(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "fixture_predictions_1008_120000.csv").write_text("sample,source_segment_idx,window_center_s,pred_range_km,true_range_km,abs_error_km\n0,5,4.5,2.25,2.0,0.25\n")

    def test_neural_commands_use_fresh_training_and_explicit_best_checkpoint(self):
        instance = self.prepare_operation()
        model = "elm_resnet18_range"
        output = self.project / "outputs/networks_results/ELM" / model
        best = output / "train_outputs/fixture_best_1008_120000.pt"

        def command(name, argv, timeout):
            if name.startswith("train_"):
                best.parent.mkdir(parents=True)
                best.write_bytes(b"checkpoint fixture")
                history = [{"epoch": epoch, "train_loss": 1.0, "val_loss": 1.0, "train_mae_km": 1.0, "val_mae_km": 1.0, "val_rmse_km": 1.0} for epoch in range(1, self.config["training"]["epochs"] + 1)]
                worker.write_json(best.parent / "fixture_history_1008_120000.json", history)
            else:
                self.predictions(output / "test_outputs")

        with mock.patch.object(instance, "command", side_effect=command) as commands, mock.patch.object(instance, "test_rows", return_value=[(5, 4.5, 2.0)]):
            result = instance.train_predict("ELM", model)
        train, predict = [call.args[1] for call in commands.call_args_list]
        self.assertEqual(train[3:6], ["train", "--model", model])
        self.assertIn("--no-resume", train)
        self.assertEqual(train[train.index("--epochs") + 1], str(self.config["training"]["epochs"]))
        self.assertEqual(train[train.index("--device") + 1], "cuda")
        self.assertEqual(predict[predict.index("--checkpoint") + 1], str(best))
        self.assertEqual(predict[predict.index("--batch-size") + 1], str(self.config["training"]["batch_size"]))
        self.assertIn("--no-plot", predict)
        self.assertFalse(list(output.rglob("*.pdf")))
        self.assertEqual(result["checkpoint"], str(best))
        self.assertEqual(result["mae_km"], 0.25)

    def test_grnn_commands_only_build_and_predict_explicit_reference(self):
        for method in tasks.METHODS:
            with self.subTest(method=method):
                self.select(method, "GRNN")
                instance = self.prepare_operation()
                model = tasks.resolve_selection(method, "GRNN")["model"]
                output = self.project / "outputs/networks_results" / method / model
                reference = output / "reference_outputs/fixture_reference_1008_120000.pt"

                def command(name, argv, timeout):
                    if name == f"grnn_{method}_build":
                        reference.parent.mkdir(parents=True)
                        reference.write_bytes(b"reference fixture")
                    else:
                        self.predictions(output / "test_outputs")

                with mock.patch.object(instance, "command", side_effect=command) as commands, mock.patch.object(instance, "test_rows", return_value=[(5, 4.5, 2.0)]):
                    result = instance.grnn(method)
                self.assertEqual([call.args[0] for call in commands.call_args_list],
                                 [f"grnn_{method}_build", f"grnn_{method}_predict"])
                build, predict = [call.args[1] for call in commands.call_args_list]
                self.assertEqual(build[3], "build")
                self.assertEqual(predict[3], "predict")
                self.assertIn("--cv-folds", build)
                self.assertIn("--sigma-candidates", build)
                for argv in (build, predict):
                    self.assertNotIn("--epochs", argv)
                    self.assertNotIn("train", argv)
                    self.assertNotIn("--model", argv)
                    self.assertEqual(argv[argv.index("--method") + 1], method)
                    self.assertEqual(argv[argv.index("--data") + 1], instance.dataset_dirs[method])
                self.assertEqual(predict[predict.index("--reference") + 1], str(reference))
                self.assertEqual(predict[predict.index("--batch-size") + 1], str(self.config["grnn"]["batch_size"]))
                self.assertIn("--no-plot", predict)
                self.assertFalse(list(output.rglob("*.pdf")))
                self.assertEqual((result["method"], result["model"]), (method, model))
                self.assertEqual(result["reference"], str(reference))
                self.assertEqual(result["mae_km"], 0.25)


if __name__ == "__main__":
    unittest.main()
