"""Study evidence checks; only the rendering test imports numerical libraries."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.data_scale_pack.localization_networks.common.scaling import resolve_capacity
from scripts.data_scale_pack.localization_networks.workflow.config import load_config
from scripts.data_scale_pack.localization_networks.workflow.remote import DEFAULT_CONFIG
from scripts.data_scale_pack.localization_networks.workflow.tasks import model_plan
from scripts.data_scale_pack.visualization.plot_study import RENDERER_PATHS, ROOT, render_study
from scripts.data_scale_pack.visualization.results import read_predictions, summarize
from scripts.data_scale_pack.visualization.study_results import digest, load_study, split_identity, validation_selection, verify_split
from scripts.run_identity import make_run_id


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def split(rows):
    rows = sorted((int(source), float(moment), float(distance)) for source, moment, distance in rows)
    values = sorted(row[2] for row in rows)
    def quantile(p):
        where = (len(values)-1)*p
        low = int(where)
        high = min(low+1, len(values)-1)
        return values[low]+(values[high]-values[low])*(where-low)
    return {"count": len(rows), "source_segment_indices": [row[0] for row in rows],
            "window_center_s": [row[1] for row in rows], "range_km": [row[2] for row in rows],
            "identity_sha256": split_identity(rows), "time_s": {"min": min(row[1] for row in rows), "max": max(row[1] for row in rows)},
            "range_km_summary": {"min": values[0], "q25": quantile(.25), "median": quantile(.5), "q75": quantile(.75), "max": values[-1]}}


def coverage(train, heldout):
    low, high = train["range_km_summary"]["min"], train["range_km_summary"]["max"]
    count = sum(low <= value <= high for value in heldout["range_km"])
    return {"within_train_range_count": count, "within_train_range_fraction": count/heldout["count"]}


def fixture(root):
    config = load_config(DEFAULT_CONFIG)
    config["training"].update(epochs=2, base_channels=2)
    inventory = [{"path": path, "sha256": hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), "size": (ROOT/path).stat().st_size}
                 for path in RENDERER_PATHS]
    study = {"schema_version": 1, "pack": "data_scale_pack", "study_id": "test-study", "status": "PASS",
             "created_at": "2026-10-15T00:00:00Z", "updated_at": "2026-10-15T00:00:00Z",
             "plan": {"schema_version": 1, "base_config": config, "fractions": [.5, 1.0], "seeds": [2026, 2027], "method": "RBD"},
             "source_files": inventory, "source_sha256": digest(inventory, compact=True), "children": []}
    study["plan_sha256"] = digest(study["plan"], compact=True)
    data_identity = {"schema_version": 1, "pack": "data_scale_pack", "inputs": [{"path": "fixture", "size": 32, "sha256": "a"*64}]}
    data_identity["cache_key"] = digest(data_identity)
    plan = model_plan("selected", {"method": "RBD", "network": None})
    for seed in study["plan"]["seeds"]:
        for fraction in study["plan"]["fractions"]:
            effective = copy.deepcopy(config)
            effective["validation"]["seed"] = seed
            effective["training"]["train_fraction"] = fraction
            filters = {"method": "RBD", "network": None}
            run_id = make_run_id("data_scale_pack", effective, filters)
            run = root/"test-study"/"children"/run_id
            request = {"schema_version": 1, "pack": "data_scale_pack", "run_id": run_id, "suite": "selected", "selection": filters, "config": effective}
            source_hash, config_hash = study["source_sha256"], digest(effective)
            manifest = {"schema_version": 1, "pack": "data_scale_pack", "run_id": run_id, "suite": "selected", "selection": filters,
                        "files": inventory, "source_sha256": source_hash, "config_sha256": config_hash}
            report = {"schema_version": 1, "pack": "data_scale_pack", "run_id": run_id, "suite": "selected", "selection": filters,
                      "purpose": "configured_model_selection_without_accuracy_acceptance_threshold", "status": "PASS", "model_plan": plan,
                      "source_sha256": source_hash, "config_sha256": config_hash, "uv_lock_sha256": "f"*64,
                      "dataset_cache": {"cache_key": data_identity["cache_key"]}, "checks": [], "metrics": []}
            archive = {"schema_version": 1, "pack": "data_scale_pack", "run_id": run_id, "models": []}
            count = int(8*fraction)
            train = split((i, float(i), .1*i) for i in range(1, count+1))
            val = split([(9, 9., .9), (10, 10., 1.)])
            test = split([(20, 20., 2.), (21, 21., 3.)])
            members = {"schema_version": 1, "train": train, "validation": val, "validation_coverage": coverage(train, val)}
            for index, job in enumerate(plan):
                profile = "auto" if job["architecture"] == "complex_CNN" else job["architecture"].removeprefix("complex_CNN_")
                scaling = json.loads(json.dumps({**resolve_capacity(profile, train_count=count, base_channels=2, scaling_policy=effective["scaling"]), "calibration_status": "uncalibrated"}))
                sampling = {"version": "temporal-block-prefix-v1", "train_fraction": fraction, "candidate_train_count": 8,
                            "train_count": count, "validation_count": 2, "block_rows": 32, "seed": seed,
                            "source_identity_sha256": hashlib.sha256(",".join(str(i) for i in range(1,count+1)).encode()).hexdigest()}
                architecture = {"pack": "data_scale_pack", "method": "RBD", "model": job["model"], "scaling": scaling, "sampling": sampling,
                                "training_inventory": members,
                                "model_config": {**{key: value for key,value in scaling.items() if key != "calibration_status"}, "input_elements": 21,
                                                 "input_freq_bins": 22, "dropout": .15, "base_channels": 2}}
                directory = run/"project/outputs/networks_results/RBD"/job["model"]
                architecture_path = directory/"train_outputs/fixture_architecture_1015_010203.json"
                write(architecture_path, architecture)
                history = []
                val_score = [0.5,0.4,0.3,0.2,0.1,0.35][index] + (seed-2026)*.01
                for epoch in (1,2):
                    history.append({"epoch": epoch, "loss_space": "km", "lr": .001, "train_loss": .01,
                                    "train_mae_km": .08, "train_rmse_km": .1,
                                    "val_mae_km": val_score, "val_rmse_km": val_score+.05*(2-epoch), "val_loss": .02,
                                    "epoch_elapsed_s": .1, "optimizer_steps_epoch": 1, "optimizer_steps": epoch})
                write(directory/"train_outputs/fixture_history_1015_010203.json", history)
                # Deliberately reverse the test ranking: validation selection must still pick xlarge.
                error = .1*(index+1)
                csv = ("sample,source_segment_idx,window_center_s,pred_range_km,true_range_km,abs_error_km\n"
                       f"0,20,20.0,{2+error:.6f},2.0,{error:.6f}\n1,21,21.0,{3+error:.6f},3.0,{error:.6f}\n").encode()
                output = directory/"test_outputs/fixture_predictions_1015_010203.csv"
                output.parent.mkdir(parents=True)
                output.write_bytes(csv)
                training = {"best_epoch": 2, "best_val_rmse_km": val_score, "best_val_mae_km": val_score,
                            "best_train_loss": .01, "best_train_mae_km": .08, "best_train_rmse_km": .1,
                            "total_train_wall_s": .4, "sum_epoch_elapsed_s": .2, "optimizer_steps": 2,
                            "parameter_count": scaling["parameter_count"], "train_identity_sha256": train["identity_sha256"],
                            "val_identity_sha256": val["identity_sha256"], "test_identity_sha256": test["identity_sha256"]}
                metric = {**summarize(read_predictions(csv)), **training, "method": "RBD", "model": job["model"], "epochs": 2,
                          "scaling": scaling, "train_fraction": fraction, "architecture_path": architecture_path.relative_to(run).as_posix(),
                          "training_inventory_path": "reports/training_inventory.json"}
                report["metrics"].append(metric)
                report["checks"].append({"name": f"evaluate_RBD_{job['model']}", "status": "PASS"})
                archive["models"].append({"method": "RBD", "model": job["model"], "seed": seed, "train_fraction": fraction,
                                          "train_indices": list(range(count)), "val_indices": [8,9], "sampling": sampling,
                                          "source_identity": {"version": 1, "train_sources": [{"path": "/fixture/train.h5", "size_bytes": 32, "sha256": "a"*64}],
                                                              "validation_sources": [{"path": "/fixture/train.h5", "size_bytes": 32, "sha256": "a"*64}]},
                                          "training_inventory": members, "test": test, "test_coverage": coverage(train,test), "training_metrics": training})
            for path,value in (("request.json",request),("manifest.json",manifest),("report.json",report),("reports/training_inventory.json",archive),("data_identity.json",data_identity)):
                write(run/path,value)
            study["children"].append({"run_id": run_id, "seed": seed, "train_fraction": fraction, "config_sha256": config_hash, "status": "PASS"})
    path = root/"test-study/study.json"
    write(path,study)
    return path,study


class StudyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="swellex-study-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path,self.study = fixture(self.root)

    def test_complete_study_uses_real_counts_and_sample_sd(self):
        data = load_study(self.path)
        self.assertEqual(data.status,"PASS")
        self.assertEqual(len(data.rows),24)
        self.assertEqual({row["train_count"] for row in data.rows},{4,8})
        self.assertEqual(len(data.summaries),12)
        self.assertTrue(all(row["n_seeds"]==2 for row in data.summaries))
        self.assertAlmostEqual(data.summaries[0]["best_val_rmse_km_sd"], .01/(2**.5))
        self.assertEqual(len(data.histories),48)

    def test_missing_and_failed_children_stay_explicit_not_run(self):
        self.study["children"].pop()
        self.study["children"][0].update(status="FAIL",error="fixture failed")
        write(self.path,self.study)
        data = load_study(self.path)
        self.assertEqual(data.status,"PARTIAL")
        self.assertEqual(len([row for row in data.rows if row["status"]=="NOT_RUN"]),12)
        self.assertEqual({row["run_status"] for row in data.rows if row["status"]=="NOT_RUN"},{"FAIL","NOT_RUN"})
        self.assertTrue(all(row["status"]=="NOT_RUN" for row in validation_selection(data)))

    def test_validation_selection_never_uses_test_ranking(self):
        data = load_study(self.path)
        for selected in validation_selection(data):
            self.assertEqual(selected["selected_architecture"],"complex_CNN_xlarge")
            self.assertGreater(selected["selected_test_rmse_km_mean"], selected["auto_test_rmse_km_mean"])
            self.assertIn("auto_minus_selected_test_rmse_km_mean",selected)

    def test_tampered_plan_hash_is_rejected(self):
        self.study["plan"]["fractions"][0]=.25
        write(self.path,self.study)
        with self.assertRaisesRegex(ValueError,"plan hash"):
            load_study(self.path)

    def test_child_source_drift_is_rejected(self):
        child=self.path.parent/"children"/self.study["children"][0]["run_id"]
        report=json.loads((child/"report.json").read_text())
        report["source_sha256"]="b"*64
        write(child/"report.json",report)
        with self.assertRaisesRegex(ValueError,"Source inventory hash"):
            load_study(self.path)

    def test_best_checkpoint_summary_cannot_be_relabelled(self):
        child=self.path.parent/"children"/self.study["children"][0]["run_id"]
        report=json.loads((child/"report.json").read_text())
        report["metrics"][0]["best_epoch"]=1
        write(child/"report.json",report)
        with self.assertRaisesRegex(ValueError,"minimum-validation-RMSE"):
            load_study(self.path)

    def test_corrupted_physical_inventory_hash_is_rejected(self):
        value=split([(1,1.,2.),(2,2.,3.)])
        value["window_center_s"][0]=8.
        with self.assertRaisesRegex(ValueError,"identity hash"):
            verify_split(value,"test")

    def rewrite_members(self, child, target, members):
        run = self.path.parent/"children"/child["run_id"]
        archive = json.loads((run/"reports/training_inventory.json").read_text())
        report = json.loads((run/"report.json").read_text())
        metric_map = {row["model"]: row for row in report["metrics"]}
        for entry in archive["models"]:
            metric = metric_map[entry["model"]]
            sidecar_path = run/metric["architecture_path"]
            sidecar = json.loads(sidecar_path.read_text())
            inventory = entry["training_inventory"]
            inventory[target] = members
            inventory["validation_coverage"] = coverage(inventory["train"],inventory["validation"])
            entry["test_coverage"] = coverage(inventory["train"],entry["test"])
            sidecar["training_inventory"] = copy.deepcopy(inventory)
            field = "train_identity_sha256" if target == "train" else "val_identity_sha256"
            metric[field] = entry["training_metrics"][field] = members["identity_sha256"]
            write(sidecar_path,sidecar)
        write(run/"report.json",report)
        write(run/"reports/training_inventory.json",archive)

    def test_same_seed_validation_membership_must_stay_fixed(self):
        child = next(row for row in self.study["children"] if row["seed"]==2026 and row["train_fraction"]==1)
        self.rewrite_members(child,"validation",split([(9,9.,.9),(11,11.,1.1)]))
        with self.assertRaisesRegex(ValueError,"Same-seed validation"):
            load_study(self.path)

    def test_training_subsets_must_be_physically_nested(self):
        child = next(row for row in self.study["children"] if row["seed"]==2026 and row["train_fraction"]==1)
        self.rewrite_members(child,"train",split((index,float(index),index*.1) for index in (1,2,3,5,6,7,8,11)))
        with self.assertRaisesRegex(ValueError,"not physically nested"):
            load_study(self.path)

    def test_render_emits_csv_json_pdf_png_and_integrity_manifest(self):
        destination=self.root/"rendered"
        manifest=render_study(self.path,destination,dpi=72)
        self.assertEqual(manifest["status"],"PASS")
        self.assertEqual(len(manifest["figures"]),12)
        self.assertTrue((destination/"validation-selection.csv").is_file())
        self.assertTrue((destination/"learning-curves-RBD.png").is_file())
        for record in manifest["files"]:
            content=(destination/record["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(),record["sha256"])
        with self.assertRaisesRegex(ValueError,"new directory"):
            render_study(self.path,destination,dpi=72)


if __name__=="__main__":
    unittest.main()
