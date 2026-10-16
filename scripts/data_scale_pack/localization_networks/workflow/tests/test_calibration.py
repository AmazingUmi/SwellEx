"""Synthetic stdlib fixtures; no original measurements are fitted in these tests."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.data_scale_pack.localization_networks.common import scaling
from scripts.data_scale_pack.localization_networks.workflow import calibration as calibration, remote, study
from scripts.run_identity import make_run_id


def synthetic_projection():
    return [{"seed": seed, "train_count": count, "profile": profile,
             "validation_rmse_km": ([1., 2., 9.][seed_index] + profile_index) * (count_index + 1),
             "parameter_count": scaling.parameter_count(profile, 16),
             "run_id": f"fixture-{seed}-{count}", "train_fraction": calibration.FRACTIONS[count_index]}
            for seed_index, seed in enumerate(calibration.DEVELOPMENT_SEEDS)
            for count_index, count in enumerate(calibration.COUNTS)
            for profile_index, profile in enumerate(calibration.PROFILES)]


class CalibrationAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_projection()

    def test_projection_is_complete_unique_validation_only_and_base16(self):
        self.assertEqual(len(calibration.validate_projection(self.rows)), 48)
        alterations = [lambda rows: rows.pop(), lambda rows: rows.__setitem__(0, copy.deepcopy(rows[1])),
                       lambda rows: rows[0].update(test_rmse_km=0.),
                       lambda rows: rows[0].update(parameter_count=42),
                       lambda rows: rows[0].update(validation_rmse_km=0),
                       lambda rows: rows[0].update(validation_rmse_km=float("nan")),
                       lambda rows: rows[0].update(validation_rmse_km=float("inf")),
                       lambda rows: rows[0].update(seed=True), lambda rows: rows[0].update(train_count=337.),
                       lambda rows: rows[0].update(profile="baseline"), lambda rows: rows[0].update(run_id="different")]
        for change in alterations:
            rows = copy.deepcopy(self.rows)
            change(rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                calibration.validate_projection(rows)

    def test_candidate_grid_is_exactly_338_unique_and_includes_original(self):
        grid = calibration.candidate_grid()
        self.assertEqual(len(grid), 338)
        self.assertIn(calibration.OLD_POLICY, grid)
        for bad in (grid[:-1], [grid[0], *grid[1:-1], grid[0]],
                    [{"reference_samples": 500, "exponent": .25}, *grid[1:]]):
            with self.assertRaises(ValueError):
                calibration.validate_grid(bad)

    def test_objective_is_ratio_of_seed_means_with_equal_count_weights(self):
        result = calibration.score_policy(self.rows, calibration.OLD_POLICY)
        # Every N has seed-mean oracle 4*(N-index+1), and each profile adds
        # profile-index*(N-index+1). Average relative regret is index/4 per N.
        expected = math.fsum(calibration.PROFILES.index(row["profile"]) / 4 for row in result["assignments"]) / 4
        self.assertAlmostEqual(result["score"], expected, places=14)
        self.assertEqual([row["oracle_profile"] for row in result["assignments"]], ["small"] * 4)
        weighted_by_count = sum(row["relative_regret"] * row["train_count"] for row in result["assignments"]) / sum(calibration.COUNTS)
        self.assertNotAlmostEqual(result["score"], weighted_by_count)

    @staticmethod
    def candidate(score, parameters, margin, exponent, reference, mapping=("small", "medium", "large", "xlarge")):
        return {"score": score, "mean_parameter_count": parameters, "threshold_margin": margin,
                "policy": {"exponent": exponent, "reference_samples": reference},
                "assignments": [{"profile": profile} for profile in mapping]}

    def test_tolerance_then_parameter_count_then_same_mapping_margin(self):
        c = self.candidate
        first = c(.1, 20, 100, .25, 600)
        second = c(.1 + 5e-13, 10, .1, .5, 700)
        third = c(.1 + 6e-13, 10, .2, 1., 1000)
        outside = c(.1 + 2e-12, 1, 1000, .25, 600)
        self.assertIs(calibration.choose_candidate([first, second, third, outside]), third)

    def test_margin_compares_only_same_mapping_and_final_tie_uses_exponent_reference(self):
        c = self.candidate
        first = c(.1, 10, 100, .5, 700)
        same_mapping = c(.1, 10, 101, .75, 1000)
        different_mapping = c(.1, 10, .001, .25, 900, ("medium",) * 4)
        lower_reference = c(.1, 10, .001, .25, 800, ("medium",) * 4)
        self.assertIs(calibration.choose_candidate([first, same_mapping, different_mapping, lower_reference]), lower_reference)

    def test_threshold_formula_and_margin_use_all_observed_counts(self):
        result = calibration.score_policy(self.rows, {"reference_samples": 900, "exponent": .5})
        self.assertAlmostEqual(result["thresholds"]["medium"], 900.)
        expected_large = 900 * (scaling.parameter_count("large", 16) / scaling.parameter_count("medium", 16)) ** 2
        self.assertAlmostEqual(result["thresholds"]["large"], expected_large, places=9)
        margin = min(abs(math.log(count / threshold)) for count in calibration.COUNTS for threshold in result["thresholds"].values())
        self.assertAlmostEqual(result["threshold_margin"], margin, places=14)

    def test_fit_is_deterministic_and_records_every_candidate(self):
        selected, candidates = calibration.fit_policy(self.rows)
        reversed_selected, reversed_candidates = calibration.fit_policy(list(reversed(self.rows)))
        self.assertEqual(selected, reversed_selected)
        self.assertEqual(candidates, reversed_candidates)
        self.assertEqual(len(candidates), 338)
        self.assertIn(selected, candidates)
        self.assertEqual(selected["score"], min(row["score"] for row in candidates))

    def test_leave_one_seed_out_never_fits_its_held_out_seed(self):
        original = calibration.leave_one_seed_out(self.rows)
        changed = copy.deepcopy(self.rows)
        for row in changed:
            if row["seed"] == 2026:
                row["validation_rmse_km"] = 100 if row["profile"] == "small" else 1
        updated = calibration.leave_one_seed_out(changed)
        self.assertEqual(original[0]["policy"], updated[0]["policy"])
        self.assertEqual(original[0]["fit_score"], updated[0]["fit_score"])
        self.assertNotEqual(original[0]["held_out"]["score"], updated[0]["held_out"]["score"])
        for fold in original:
            self.assertNotIn(fold["held_out_seed"], fold["fit_seeds"])
            self.assertEqual(len(fold["fit_seeds"]), 2)


class CalibrationProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swellex-calibration-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.analysis_path = self.root / "analysis" / "study-results.json"
        self.analysis_path.parent.mkdir()
        self.study_path = self.root / "study.json"
        self.manifest_path = self.analysis_path.parent / "manifest.json"
        config = remote.load_config(remote.DEFAULT_CONFIG)
        plan = {"schema_version": 1, "method": "SCM", "base_config": config,
                "seeds": list(calibration.DEVELOPMENT_SEEDS), "fractions": list(calibration.FRACTIONS)}
        files = [{"path": "scripts/source.py", "size": 1, "sha256": "a" * 64}]
        children = []
        metrics = []
        for seed in calibration.DEVELOPMENT_SEEDS:
            for fraction in calibration.FRACTIONS:
                child_config = study.child_config(plan, seed, fraction)
                run_id = make_run_id("data_scale_pack", child_config, {"method": "SCM", "network": None})
                children.append({"run_id": run_id, "seed": seed, "train_fraction": fraction,
                                 "config_sha256": study.digest(child_config, compact=False), "status": "PASS"})
                count = calibration.COUNTS[calibration.FRACTIONS.index(fraction)]
                for profile in calibration.PROFILES:
                    metrics.append({"study_id": "development", "run_id": run_id, "method": "SCM", "status": "PASS", "run_status": "PASS",
                                    "seed": seed, "train_fraction": fraction, "train_count": count,
                                    "architecture": "complex_CNN_" + profile, "model": "scm_complex_cnn_" + profile + "_range",
                                    "selection_mode": "fixed", "selected_profile": profile,
                                    "best_val_rmse_km": 1 + calibration.PROFILES.index(profile),
                                    "parameter_count": scaling.parameter_count(profile, 16), "test_rmse_km": 9999})
        self.ledger = {"schema_version": 1, "pack": "data_scale_pack", "study_id": "development", "status": "PASS",
                       "plan": plan, "plan_sha256": study.digest(plan), "source_files": files,
                       "source_sha256": study.digest(files), "children": children}
        self.analysis = {"schema_version": 1, "pack": "data_scale_pack", "study_id": "development", "status": "PASS",
                         "study": self.ledger, "metrics": metrics}
        self.write_fixture()

    def write_fixture(self):
        self.study_path.write_text(json.dumps(self.ledger, indent=2))
        self.analysis_path.write_text(json.dumps(self.analysis, indent=2))
        manifest = {"schema_version": 1, "pack": "data_scale_pack", "study_id": "development", "status": "PASS",
                    "source_sha256": self.ledger["source_sha256"], "plan_sha256": self.ledger["plan_sha256"],
                    "files": [{"path": self.analysis_path.name, "size": self.analysis_path.stat().st_size,
                               "sha256": hashlib.sha256(self.analysis_path.read_bytes()).hexdigest()}],
                    "inputs": [{"path": str(self.study_path), "size": self.study_path.stat().st_size,
                                "sha256": hashlib.sha256(self.study_path.read_bytes()).hexdigest()}]}
        self.manifest_path.write_text(json.dumps(manifest))

    def test_load_projects_exactly_48_fixed_validation_observations(self):
        rows, source = calibration.load_development_projection(self.analysis_path)
        self.assertEqual(len(rows), 48)
        self.assertTrue(all(set(row) == calibration.PROJECTION_FIELDS for row in rows))
        self.assertEqual(source["training_source_sha256"], self.ledger["source_sha256"])
        self.assertEqual(source["study"]["sha256"], hashlib.sha256(self.study_path.read_bytes()).hexdigest())

    def test_test_metrics_and_automatic_baseline_rows_do_not_enter_projection(self):
        before, _ = calibration.load_development_projection(self.analysis_path)
        for row in self.analysis["metrics"]:
            row["test_rmse_km"] = -100000
        self.analysis["metrics"].extend([{"architecture": "complex_CNN", "best_val_rmse_km": 0},
                                         {"architecture": "complex_CNN_baseline", "best_val_rmse_km": 0}])
        self.write_fixture()
        after, _ = calibration.load_development_projection(self.analysis_path)
        self.assertEqual(before, after)
        self.assertEqual(study.digest(before), study.digest(after))

    def test_manifest_raw_file_and_source_study_hashes_are_enforced(self):
        self.analysis_path.write_text(self.analysis_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "file bytes"):
            calibration.load_development_projection(self.analysis_path)
        self.write_fixture()
        self.study_path.write_text(self.study_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "source study hash"):
            calibration.load_development_projection(self.analysis_path)
        self.write_fixture()
        manifest = json.loads(self.manifest_path.read_text())
        manifest["source_sha256"] = "b" * 64
        self.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "hashes"):
            calibration.load_development_projection(self.analysis_path)

    def test_partial_duplicate_or_crossed_source_observations_are_rejected(self):
        self.ledger["children"][0]["status"] = "FAIL"
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "must be PASS"):
            calibration.load_development_projection(self.analysis_path)
        self.ledger["children"][0]["status"] = "PASS"
        self.analysis["metrics"][0]["run_id"] = self.ledger["children"][1]["run_id"]
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "source child"):
            calibration.load_development_projection(self.analysis_path)

    def test_synthetic_cli_outputs_artifact_projection_hash_and_338_score_rows(self):
        output = self.root / "synthetic-output"
        result = calibration.calibrate(self.analysis_path, output)
        artifact_path = output / calibration.ARTIFACT_NAME
        self.assertEqual(json.loads(artifact_path.read_text()), result)
        self.assertEqual(result["projection_sha256"], study.digest(result["projection"]))
        self.assertEqual(result["development_seeds"], [2026, 2027, 2028])
        self.assertFalse(result["objective"]["test_metrics_used"])
        self.assertEqual(len(result["leave_one_seed_out"]), 3)
        with (output / "candidate-scores.csv").open(newline="") as handle:
            scores = list(csv.DictReader(handle))
        self.assertEqual(len(scores), 338)
        self.assertEqual(sum(row["selected"] == "True" for row in scores), 1)
        self.assertEqual(result["grid"]["candidate_scores"]["sha256"], hashlib.sha256((output / "candidate-scores.csv").read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, "new directory"):
            calibration.calibrate(self.analysis_path, output)


if __name__ == "__main__":
    unittest.main()
