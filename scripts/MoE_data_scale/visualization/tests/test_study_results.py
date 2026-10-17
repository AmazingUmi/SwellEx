"""Standard-library fixtures for evidence identities and incomplete matrices."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.MoE_data_scale.localization_networks.common.scaling import resolve_capacity, serialized_resolution_metadata
from scripts.MoE_data_scale.localization_networks.workflow.config import load_config
from scripts.MoE_data_scale.localization_networks.workflow.study import child_config, selection_for
from scripts.MoE_data_scale.localization_networks.workflow.tasks import model_plan
from scripts.MoE_data_scale.visualization.results import load_run, read_predictions, summarize
from scripts.MoE_data_scale.visualization.study_results import digest, load_study, split_identity, summarize_study, validation_selection
from scripts.run_identity import make_run_id


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def split(rows):
    rows = sorted((int(i), float(t), float(r)) for i, t, r in rows)
    values = sorted(row[2] for row in rows)
    def quantile(p):
        where = (len(values) - 1) * p
        low = int(where)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (where - low)
    return {"count": len(rows), "source_segment_indices": [row[0] for row in rows],
            "window_center_s": [row[1] for row in rows], "range_km": [row[2] for row in rows],
            "identity_sha256": split_identity(rows), "time_s": {"min": rows[0][1], "max": rows[-1][1]},
            "range_km_summary": {"min": values[0], "q25": quantile(.25), "median": quantile(.5), "q75": quantile(.75), "max": values[-1]}}


def signature():
    blocks = []
    offset = 0
    for method, height, pairs in (("RBD", 2, []), ("SCM", 3, [[1, 1], [1, 2], [2, 2]])):
        physical = {"method": method, "input_shape": [2, height, 2], "frequency_hz": [100., 200.],
                    "element_idx": [1, 2], "depth_m": [10., 20.], "pair_indices": pairs}
        blocks.append({"method": method, "start": offset, "stop": offset + height, "input_signature": physical})
        offset += height
    return {"format_version": 1, "method": "RBD_SCM", "input_shape": [2, 5, 2], "fusion_strategy": "block_rms", "blocks": blocks,
            "source_contract": {"input_files": [{"path": "/raw/fixture.mat", "size_bytes": 32, "mtime_ns": 1, "sha256": "a" * 64}],
                                "effective_config": {"dataset": {"num_snapshots_per_segment": 1, "snapshot_overlap_count": 0}},
                                "split": {"strategy": "range_overlap"}, "generator_version": "fixture"}}


def fixture(root):
    config = load_config(Path("scripts/MoE_data_scale/localization_networks/workflow/haiqin_range_overlap_train_before.toml"))
    config["training"].update(epochs=2, base_channels=2, batch_size=4)
    files = [{"path": "scripts/fixture.py", "size": 1, "sha256": "b" * 64}]
    plan = {"schema_version": 1, "study_id": "test-moe-study", "base_config": config,
            "method": "RBD_SCM", "fractions": [.5, 1.0], "experts": [1, 2], "seeds": [2026, 2027]}
    study = {"schema_version": 1, "pack": "MoE_data_scale", "study_id": plan["study_id"], "status": "PASS",
             "plan": plan, "plan_sha256": digest(plan, compact=True), "source_files": files,
             "source_sha256": digest(files, compact=True), "children": []}
    data_identity = {"schema_version": 1, "pack": "MoE_data_scale", "dataset": config["dataset"], "rbd": config["rbd"], "uv_lock_sha256": "f" * 64,
                     "generator_sources": [], "validator_sources": [],
                     "inputs": [{"path": "fixture", "size": 32, "sha256": "a" * 64}]}
    data_identity["cache_key"] = digest(data_identity)
    for seed in plan["seeds"]:
        for fraction in plan["fractions"]:
            for experts in plan["experts"]:
                config = child_config(plan, seed, fraction, experts)
                suite, selection = selection_for(plan, experts)
                jobs = model_plan(suite, selection)
                job = jobs[0]
                run_id = make_run_id("MoE_data_scale", config, selection)
                run = root / plan["study_id"] / "children" / run_id
                count = int(8 * fraction)
                train = split((i, i, .1 * i) for i in range(1, count + 1))
                val = split([(9, 9., .9), (10, 10., 1.)])
                test = split([(20, 20., 2.), (21, 21., 3.)])
                inventory = {"schema_version": 1, "train": train, "validation": val}
                resolved = resolve_capacity(experts, train_count=count, base_channels=2, scaling_policy=config["scaling"])
                model_config = json.loads(json.dumps({**resolved, "input_features": 5, "input_freq_bins": 2, "dropout": .15, "base_channels": 2}))
                scaling = serialized_resolution_metadata(model_config, "input_features")
                sampling = {"version": "temporal-block-prefix-v1", "train_fraction": fraction, "candidate_train_count": 8,
                            "train_count": count, "validation_count": 2, "block_rows": 32, "seed": seed,
                            "source_identity_sha256": hashlib.sha256(",".join(map(str, range(1, count + 1))).encode()).hexdigest()}
                prep = {"strategy": "block_rms", "scales": [.5, 1.5], "fit_sample_count": count,
                        "fit_source_identity_sha256": sampling["source_identity_sha256"]}
                architecture = {"pack": "MoE_data_scale", "method": "RBD_SCM", "model": job["model"], "scaling": scaling,
                                "sampling": sampling, "training_inventory": inventory, "model_config": model_config,
                                "balance_loss_weight": config["moe"]["balance_loss_weight"],
                                "input_preprocessing": prep, "input_signature": signature()}
                directory = run / "project/outputs/networks_results/RBD_SCM" / job["model"]
                arch_path = directory / "train_outputs/fixture_architecture_1016_010203.json"
                write(arch_path, architecture)
                score = .5 - experts * .1 + (seed - 2026) * .01
                history = []
                for epoch in (1, 2):
                    row = {"epoch": epoch, "loss_space": "km", "lr": .001, "epoch_elapsed_s": .1,
                           "optimizer_steps_epoch": count // 4, "optimizer_steps": epoch * (count // 4)}
                    for name, samples in (("train", count), ("val", 2)):
                        balance = 0. if experts == 1 else .1
                        row.update({f"{name}_loss": .1, f"{name}_total_loss": .1 + architecture["balance_loss_weight"] * balance,
                                    f"{name}_balance_loss": balance, f"{name}_router_entropy": 0. if experts == 1 else .5,
                                    f"{name}_expert_load": [samples] * experts, f"{name}_expert_importance": [1 / experts] * experts,
                                    f"{name}_mae_km": score, f"{name}_rmse_km": score + (2 - epoch) * .01})
                    history.append(row)
                write(directory / "train_outputs/fixture_history_1016_010203.json", history)
                # Test ranks E1 better; validation ranks E2 better.
                error = .1 * experts
                csv = ("sample,source_segment_idx,window_center_s,pred_range_km,true_range_km,abs_error_km\n"
                       f"0,20,20.0,{2 + error:.6f},2.0,{error:.6f}\n1,21,21.0,{3 + error:.6f},3.0,{error:.6f}\n").encode()
                out = directory / "test_outputs/fixture_predictions_1016_010203.csv"
                out.parent.mkdir(parents=True)
                out.write_bytes(csv)
                training = {"best_epoch": 2, "best_val_rmse_km": score, "best_val_mae_km": score,
                            "best_train_loss": .1, "best_train_mae_km": score, "best_train_rmse_km": score,
                            "total_train_wall_s": .3, "sum_epoch_elapsed_s": .2, "optimizer_steps": count // 2,
                            "parameter_count": scaling["parameter_count"], "train_identity_sha256": train["identity_sha256"],
                            "val_identity_sha256": val["identity_sha256"], "test_identity_sha256": test["identity_sha256"]}
                metric = {**summarize(read_predictions(csv)), **training, "method": job["method"], "model": job["model"], "epochs": 2,
                          "scaling": scaling, "train_fraction": fraction, "architecture_path": arch_path.relative_to(run).as_posix(),
                          "training_inventory_path": "reports/training_inventory.json", "input_preprocessing": prep}
                common = {"schema_version": 1, "pack": "MoE_data_scale", "run_id": run_id, "suite": suite, "selection": selection}
                report = {**common, "purpose": "configured_model_selection_without_accuracy_acceptance_threshold", "status": "PASS",
                          "model_plan": jobs, "source_sha256": study["source_sha256"], "config_sha256": digest(config), "uv_lock_sha256": "f" * 64,
                          "dataset_cache": {"cache_key": data_identity["cache_key"]}, "checks": [{"name": f"evaluate_{job['method']}_{job['model']}", "status": "PASS"}],
                          "metrics": [metric]}
                entry = {"method": job["method"], "model": job["model"], "seed": seed, "train_fraction": fraction,
                         "train_indices": list(range(count)), "val_indices": [8, 9], "sampling": sampling,
                         "source_identity": {"version": 1, "train_sources": [{"path": str(run / "train.h5"), "size_bytes": 32, "sha256": "a" * 64}],
                                             "validation_sources": [{"path": str(run / "train.h5"), "size_bytes": 32, "sha256": "a" * 64}]},
                         "training_inventory": inventory, "test": test, "training_metrics": training, "input_preprocessing": prep}
                records = {"report.json": report, "request.json": {**common, "config": config},
                           "manifest.json": {**common, "files": files, "source_sha256": study["source_sha256"], "config_sha256": digest(config)},
                           "data_identity.json": data_identity,
                           "reports/training_inventory.json": {"schema_version": 1, "pack": "MoE_data_scale", "run_id": run_id, "models": [entry]}}
                for path, value in records.items():
                    write(run / path, value)
                study["children"].append({"run_id": run_id, "seed": seed, "train_fraction": fraction, "experts": experts,
                                          "network": selection["network"], "model": job["model"], "config_sha256": digest(config), "status": "PASS"})
    path = root / plan["study_id"] / "study.json"
    write(path, study)
    return path, study


class StudyResultsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="moe-study-results-")
        self.addCleanup(temporary.cleanup)
        self.path, self.study = fixture(Path(temporary.name))

    def child(self, index=0):
        return self.path.parent / "children" / self.study["children"][index]["run_id"]

    def mutate_split(self, index, name, value):
        child = self.child(index)
        report_path = child / "report.json"
        report = json.loads(report_path.read_text())
        row = report["metrics"][0]
        archive_path = child / "reports/training_inventory.json"
        archive = json.loads(archive_path.read_text())
        entry = archive["models"][0]
        sidecar_path = child / row["architecture_path"]
        sidecar = json.loads(sidecar_path.read_text())
        key = "train_identity_sha256" if name == "train" else "val_identity_sha256"
        row[key] = entry["training_metrics"][key] = value["identity_sha256"]
        entry["training_inventory"][name] = copy.deepcopy(value)
        sidecar["training_inventory"][name] = copy.deepcopy(value)
        for path, contents in ((report_path, report), (archive_path, archive), (sidecar_path, sidecar)):
            write(path, contents)

    def test_validation_identity_fixed_within_seed(self):
        self.mutate_split(2, "validation", split([(12, 12., 1.2), (13, 13., 1.3)]))
        with self.assertRaisesRegex(ValueError, "Same-seed validation"):
            load_study(self.path)

    def test_same_fraction_different_experts_require_same_training(self):
        self.mutate_split(1, "train", split([(2, 2., .2), (3, 3., .3), (4, 4., .4), (11, 11., 1.1)]))
        with self.assertRaisesRegex(ValueError, "Same-seed/fraction expert training"):
            load_study(self.path)

    def test_fractions_require_nested_physical_training(self):
        changed = split((i, i, .1 * i) for i in [2, 3, 4, 5, 6, 7, 8, 11])
        for index in (2, 3):
            self.mutate_split(index, "train", changed)
        with self.assertRaisesRegex(ValueError, "not physically nested"):
            load_study(self.path)

    def test_same_fraction_requires_same_fitted_scales_across_experts(self):
        child = self.child(1)
        report_path = child / "report.json"
        archive_path = child / "reports/training_inventory.json"
        report = json.loads(report_path.read_text())
        archive = json.loads(archive_path.read_text())
        sidecar_path = child / report["metrics"][0]["architecture_path"]
        sidecar = json.loads(sidecar_path.read_text())
        for record in (report["metrics"][0], archive["models"][0], sidecar):
            record["input_preprocessing"]["scales"][0] = .6
        for path, value in ((report_path, report), (archive_path, archive), (sidecar_path, sidecar)):
            write(path, value)
        with self.assertRaisesRegex(ValueError, "different fitted block_rms"):
            load_study(self.path)

    def test_reject_cross_cell_raw_input_drift(self):
        child = self.child(1)
        identity = json.loads((child / "data_identity.json").read_text())
        identity["inputs"][0]["sha256"] = "d" * 64
        identity["cache_key"] = digest({key: value for key, value in identity.items() if key != "cache_key"})
        report = json.loads((child / "report.json").read_text())
        report["dataset_cache"]["cache_key"] = identity["cache_key"]
        write(child / "data_identity.json", identity)
        write(child / "report.json", report)
        with self.assertRaisesRegex(ValueError, "different raw inputs"):
            load_study(self.path)

    def test_single_run_joint_label_and_physical_axes(self):
        from scripts.MoE_data_scale.visualization.plot_results import prediction_grid_layout
        data = load_run(self.child())
        self.assertIn("RBD_SCM/block_rms", data.models[0].label)
        self.assertEqual(prediction_grid_layout(data.models), [[0]])
        self.assertEqual(data.models[0].architecture["model_config"]["input_features"], 5)
        sidecar_path = self.child() / data.report["metrics"][0]["architecture_path"]
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["input_signature"]["blocks"][1]["input_signature"]["pair_indices"][0] = [2, 1]
        write(sidecar_path, sidecar)
        with self.assertRaisesRegex(ValueError, "pair axes"):
            load_run(self.child())

    def test_single_run_rejects_unfitted_or_missing_block_rms_state(self):
        report_path = self.child() / "report.json"
        report = json.loads(report_path.read_text())
        path = self.child() / report["metrics"][0]["architecture_path"]
        original = json.loads(path.read_text())
        for key, value in (("fit_sample_count", 5), ("fit_source_identity_sha256", "c" * 64), ("scales", [0., 1.])):
            with self.subTest(field=key):
                changed = copy.deepcopy(original)
                changed["input_preprocessing"][key] = value
                write(path, changed)
                with self.assertRaises(ValueError):
                    load_run(self.child())
        changed = copy.deepcopy(original)
        del changed["input_preprocessing"]
        write(path, changed)
        with self.assertRaises(ValueError):
            load_run(self.child())

    def test_complete_matrix_and_sample_sd(self):
        data = load_study(self.path)
        self.assertEqual(data.status, "PASS")
        self.assertEqual(len(data.rows), 8)
        self.assertEqual(len(data.summaries), 4)
        self.assertAlmostEqual(data.summaries[0]["best_val_rmse_km_sd"], .01 / 2 ** .5)
        self.assertEqual(data.rows[0]["best_train_expert_load"], [4])
        self.assertEqual(data.rows[0]["best_val_expert_load"], [2])
        self.assertEqual(data.rows[0]["best_epoch"], 2)
        self.assertGreater(data.rows[1]["active_parameter_count"], 0)
        report = summarize_study(self.path.parent)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue((self.path.parent / "study-cells.csv").is_file())

    def test_partial_seed_group_has_no_mean_or_sd(self):
        self.study["children"].pop()
        write(self.path, self.study)
        data = load_study(self.path)
        self.assertEqual(data.status, "PARTIAL")
        incomplete = [row for row in data.summaries if row["status"] != "PASS"]
        self.assertEqual(len(incomplete), 1)
        self.assertIsNone(incomplete[0]["best_val_rmse_km_mean"])
        self.assertIsNone(incomplete[0]["best_val_rmse_km_sd"])
        self.assertEqual(incomplete[0]["missing_seeds"], [2027])
        report = summarize_study(self.path.parent)
        self.assertEqual(len(report["missing_cells"]), 1)

    def test_selection_uses_validation_instead_of_test(self):
        selected = validation_selection(load_study(self.path))
        self.assertEqual([row["selected_experts"] for row in selected], [2, 2])
        self.assertTrue(all(row["test_used_for_selection"] is False for row in selected))

    def test_tampered_plan_rejected(self):
        self.study["plan"]["fractions"][0] = .25
        write(self.path, self.study)
        with self.assertRaisesRegex(ValueError, "plan hash"):
            load_study(self.path)

    def test_tampered_child_source_and_checkpoint_rejected(self):
        report_path = self.child() / "report.json"
        report = json.loads(report_path.read_text())
        report["source_sha256"] = "c" * 64
        write(report_path, report)
        with self.assertRaisesRegex(ValueError, "Source inventory hash"):
            load_study(self.path)
        partial = summarize_study(self.path.parent)
        self.assertEqual(partial["status"], "PARTIAL")
        self.assertTrue(partial["validation_errors"])
        self.assertEqual(partial["summaries"], [])

    def test_best_epoch_cannot_use_test_checkpoint(self):
        path = self.child() / "report.json"
        report = json.loads(path.read_text())
        report["metrics"][0]["best_epoch"] = 1
        write(path, report)
        with self.assertRaisesRegex(ValueError, "best epoch"):
            load_study(self.path)


if __name__ == "__main__":
    unittest.main()
