"""Synthetic HDF5/stage checks executed only by the explicit haiqin CUDA suite."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from scripts.MoE_learn.localization_networks.data import load_prepared_data, stage_memberships
from scripts.MoE_learn.localization_networks.rbd_scm.network.data import JointRangeH5Dataset


def signature():
    axes = {"frequency_hz": [100., 105., 110.], "element_idx": [1, 2], "depth_m": [100., 110.]}
    rbd = dict(deepcopy(axes), method="RBD", input_shape=[2, 2, 3], pair_indices=[])
    scm = dict(deepcopy(axes), method="SCM", input_shape=[2, 3, 3], pair_indices=[[1, 1], [1, 2], [2, 2]])
    return {"format_version": 1, "method": "RBD_SCM", "input_shape": [2, 5, 3],
            "fusion_strategy": "block_rms",
            "blocks": [{"method": "RBD", "start": 0, "stop": 2, "input_signature": rbd},
                       {"method": "SCM", "start": 2, "stop": 5, "input_signature": scm}],
            "source_contract": {
                "input_files": [{"path": "synthetic.dat", "size_bytes": 12, "mtime_ns": 1, "sha256": "a" * 64}],
                "effective_config": {"dataset": {"num_snapshots_per_segment": 1, "snapshot_overlap_count": 0}},
                "split": {"strategy": "full_interleaved"}, "generator_version": "synthetic-test-v1"}}


def write_joint(path, ids):
    ids = np.asarray(ids)
    with h5py.File(path, "w") as handle:
        values = np.broadcast_to(ids[:, None, None, None], (len(ids), 5, 3, 2)).astype(np.float32).copy()
        values[:, 2:] *= 10
        features = handle.create_dataset("X", data=values)
        features.attrs["axis_order"] = features.attrs["storage_axis_order"] = "sample,feature,frequency,real_imag"
        handle.attrs["pack"] = "MoE_learn"
        handle.attrs["fusion_input_signature"] = json.dumps(signature())
        for key, values in {
            "y_range_km": ids / 10., "valid_sample": np.ones(len(ids)), "split/source_segment_idx": ids,
            "time/window_start_s": ids - 1., "time/window_center_s": ids - .5,
            "time/window_stop_s": ids.astype(float), "time/sample_start_idx": ids * 100 - 99,
            "time/sample_stop_idx": ids * 100,
        }.items():
            handle.create_dataset(key, data=np.asarray(values).reshape(-1, 1))


class ContinualDataContracts(unittest.TestCase):
    def setUp(self):
        requested = os.environ.get("SWELLEX_MOE_LEARN_TEST_DEVICE")
        if not requested or torch.device(requested).type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("MoE_learn data checks require explicitly selected CUDA on haiqin")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train_path, self.test_path = self.root / "train.h5", self.root / "test.h5"
        # Descending file order ensures stages use physical time rather than row order.
        write_joint(self.train_path, np.arange(18, 0, -1))
        write_joint(self.test_path, np.arange(101, 105))
        self.config = {"dataset": {"stage_mode": "interleaved", "val_fraction": .25,
                                   "split_strategy": "full_interleaved"},
                       "training": {"seed": 2026}, "continual": {"stages": 3}}

    def test_stage_memberships_are_disjoint_complete_and_time_ordered(self):
        for mode in ("interleaved", "chronological"):
            self.config["dataset"]["stage_mode"] = mode
            result = load_prepared_data(self.config, self.train_path, self.test_path)
            groups = result["metadata"]["stages"]
            seen = []
            for stage, group in enumerate(groups):
                train_ids, val_ids = group["train_source_ids"], group["val_source_ids"]
                self.assertTrue(train_ids and val_ids)
                self.assertFalse(set(train_ids) & set(val_ids))
                actual = sorted([*train_ids, *val_ids])
                expected = list(range(stage + 1, 19, 3)) if mode == "interleaved" else list(range(stage * 6 + 1, stage * 6 + 7))
                self.assertEqual(actual, expected)
                seen.extend(actual)
            self.assertEqual(sorted(seen), list(range(1, 19)))
            self.assertEqual(len(seen), len(set(seen)))
            self.assertFalse(set(seen) & set(result["metadata"]["test_source_ids"]))

    def test_only_first_stage_optimizer_rows_fit_rms(self):
        dataset = JointRangeH5Dataset([self.train_path])
        try:
            memberships = stage_memberships(dataset, 3, "interleaved", .25, 2026)
            selected = memberships[0][0]
            source_ids = [dataset.source_segment_idx[index] for index in selected]
        finally:
            dataset.close()
        with h5py.File(self.train_path, "r+") as handle:
            for index in set(range(18)) - set(selected):
                handle["X"][index] = 1.e6
        with h5py.File(self.test_path, "r+") as handle:
            handle["X"][:] = 1.e8
        result = load_prepared_data(self.config, self.train_path, self.test_path)
        state = result["metadata"]["preprocessing"]
        expected = math.sqrt(2. * sum(source ** 2 for source in source_ids) / len(source_ids))
        self.assertAlmostEqual(state["scales"][0], expected)
        self.assertAlmostEqual(state["scales"][1], expected * 10)
        self.assertEqual(state["fit_sample_count"], len(source_ids))
        self.assertEqual(state["fit_source_identity_sha256"],
                         hashlib.sha256(",".join(str(source) for source in source_ids).encode()).hexdigest())
        self.assertEqual(result["metadata"]["preprocessing_fit_stage"], 1)
        restored = JointRangeH5Dataset([self.test_path], preprocessing_state=state)
        try:
            for index in range(len(restored)):
                np.testing.assert_allclose(result["test_x"][index], restored[index][0].numpy())
        finally:
            restored.close()

    def test_train_test_axes_sources_and_pack_must_match(self):
        for mutation in ("frequency", "source", "overlap", "pack"):
            with self.subTest(mutation=mutation):
                write_joint(self.test_path, np.arange(101, 105))
                with h5py.File(self.test_path, "r+") as handle:
                    if mutation == "pack":
                        handle.attrs["pack"] = "MoE_data_scale"
                    elif mutation == "overlap":
                        handle["split/source_segment_idx"][0] = 1
                    else:
                        altered = json.loads(handle.attrs["fusion_input_signature"])
                        if mutation == "frequency":
                            for block in altered["blocks"]:
                                block["input_signature"]["frequency_hz"][-1] = 111.
                        else:
                            altered["source_contract"]["input_files"][0]["sha256"] = "b" * 64
                        handle.attrs["fusion_input_signature"] = json.dumps(altered)
                with self.assertRaises(ValueError):
                    load_prepared_data(self.config, self.train_path, self.test_path)

    def test_stage_demand_rejects_insufficient_samples(self):
        self.config["continual"]["stages"] = 7
        with self.assertRaisesRegex(ValueError, "at least three"):
            load_prepared_data(self.config, self.train_path, self.test_path)

    def _before_after_fixture(self, before_count=800, after_count=700, stages=4):
        # Physical time and sample IDs increase with IDs; file storage is reversed.
        write_joint(self.train_path, np.arange(before_count, 0, -1))
        write_joint(self.test_path, np.arange(1000 + after_count, 1000, -1))
        for path in (self.train_path, self.test_path):
            with h5py.File(path, "r+") as handle:
                value = json.loads(handle.attrs["fusion_input_signature"])
                value["source_contract"]["effective_config"]["dataset"].update(
                    split_strategy="range_overlap", train_side="before")
                value["source_contract"]["split"] = {"strategy": "range_overlap", "train_side": "before"}
                value["source_contract"]["generator_version"] = "moe-learn-python-features-v3-window-centered"
                handle.attrs["fusion_input_signature"] = json.dumps(value)
        self.config["dataset"].update(stage_mode="before_then_after", split_strategy="range_overlap", train_side="before",
                                      val_fraction=.2, after_test_fraction=.2, after_grouping="time_interleaved")
        self.config["dataset"].pop("after_range_edges_km", None)
        self.config["dataset"].pop("distance_edges_km", None)
        self.config["dataset"].pop("heldout_group", None)
        self.config["continual"]["stages"] = stages

    def test_before_after_counts_fixed_test_and_time_interleaved_increments(self):
        self._before_after_fixture()
        first = load_prepared_data(self.config, self.train_path, self.test_path)
        second = load_prepared_data(self.config, self.train_path, self.test_path)
        metadata = first["metadata"]
        self.assertEqual(metadata, second["metadata"])
        self.assertEqual(metadata["allocation"]["before"]["total_count"], 800)
        self.assertEqual(metadata["allocation"]["before"]["train_count"], 640)
        self.assertEqual(metadata["allocation"]["before"]["val_count"], 160)
        after = metadata["allocation"]["after"]
        self.assertEqual((after["total_count"], after["test_count"], after["val_count"], after["train_count"]),
                         (700, 140, 112, 448))
        self.assertEqual(len(first["test_y"]), 140)
        self.assertEqual(metadata["test_source_ids"], after["test_source_ids"])
        self.assertEqual(metadata["test_source_side"], "after")
        self.assertIn("per-window per-element mean centering", metadata["generation_caveat"])
        self.assertEqual(metadata["stages"][0]["source_side"], "before")
        self.assertEqual(metadata["stages"][0]["pool"], "before_source")
        all_ids = list(metadata["test_source_ids"])
        for number, stage in enumerate(metadata["stages"]):
            for role in ("train", "val"):
                ids = stage[f"{role}_source_ids"]
                self.assertEqual(len(first["stages"][number][f"{role}_y"]), len(ids))
                all_ids.extend(ids)
                if number:
                    self.assertEqual(ids, after[f"{role}_source_ids"][number - 1::3])
                    self.assertEqual(stage["source_side"], "after")
                    centers = stage[f"{role}_window_center_s"]
                    self.assertEqual(centers, sorted(centers))
            np.testing.assert_array_equal(first["stages"][number]["train_x"], second["stages"][number]["train_x"])
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), set(range(1, 801)) | set(range(1001, 1701)))
        for name in ("source_identity_partition", "physical_window_partition", "before_after_temporal_separation"):
            self.assertEqual(metadata["allocation"][name], "PASS")
        self.config["training"]["seed"] += 1
        changed_seed = load_prepared_data(self.config, self.train_path, self.test_path)
        self.assertNotEqual(metadata["test_source_ids"], changed_seed["metadata"]["test_source_ids"])

    def test_before_after_preprocessing_excludes_before_validation_and_all_after_roles(self):
        self._before_after_fixture(before_count=30, after_count=30)
        reference = load_prepared_data(self.config, self.train_path, self.test_path)
        first_stage = reference["metadata"]["stages"][0]
        with h5py.File(self.train_path, "r+") as handle:
            for index in first_stage["val_idx"]:
                handle["X"][index] = 1e6
        with h5py.File(self.test_path, "r+") as handle:
            handle["X"][:] = 1e8
        changed = load_prepared_data(self.config, self.train_path, self.test_path)
        self.assertEqual(reference["metadata"]["preprocessing"], changed["metadata"]["preprocessing"])
        np.testing.assert_array_equal(reference["stages"][0]["train_x"], changed["stages"][0]["train_x"])
        self.assertEqual(changed["metadata"]["preprocessing"]["fit_sample_count"], len(first_stage["train_source_ids"]))
        expected = math.sqrt(2. * sum(source ** 2 for source in first_stage["train_source_ids"]) / len(first_stage["train_source_ids"]))
        self.assertAlmostEqual(changed["metadata"]["preprocessing"]["scales"][0], expected)
        self.assertAlmostEqual(changed["metadata"]["preprocessing"]["scales"][1], expected * 10)
        restored = JointRangeH5Dataset([self.test_path], preprocessing_state=changed["metadata"]["preprocessing"])
        try:
            for stage in changed["metadata"]["stages"][1:]:
                for role in ("train", "val"):
                    for local, source_index in enumerate(stage[f"{role}_idx"]):
                        np.testing.assert_array_equal(changed["stages"][stage["stage"] - 1][f"{role}_x"][local], restored[source_index][0].numpy())
            for local, source_index in enumerate(changed["metadata"]["test_idx"]):
                np.testing.assert_array_equal(changed["test_x"][local], restored[source_index][0].numpy())
        finally:
            restored.close()

    def test_before_after_rejects_id_time_sample_overlap_and_reversed_sides(self):
        for mutation in ("source_id", "time", "sample", "reversed_sides"):
            with self.subTest(mutation=mutation):
                self._before_after_fixture(before_count=30, after_count=30)
                if mutation == "reversed_sides":
                    with h5py.File(self.train_path, "r+") as handle:
                        for name in ("window_start_s", "window_center_s", "window_stop_s"):
                            handle[f"time/{name}"][:] += 10000
                else:
                    with h5py.File(self.test_path, "r+") as handle:
                        if mutation == "source_id":
                            handle["split/source_segment_idx"][0] = 1
                        elif mutation == "time":
                            for name, value in (("window_start_s", 0.), ("window_center_s", .5), ("window_stop_s", 1.)):
                                handle[f"time/{name}"][0] = value
                        else:
                            handle["time/sample_start_idx"][0] = 1
                            handle["time/sample_stop_idx"][0] = 100
                with self.assertRaises(ValueError):
                    load_prepared_data(self.config, self.train_path, self.test_path)

    def test_before_after_constraints_and_nonempty_increments(self):
        for mutation in ("split_strategy", "train_side", "source_provenance", "legacy_generator", "insufficient_after"):
            with self.subTest(mutation=mutation):
                self._before_after_fixture(before_count=30, after_count=4 if mutation == "insufficient_after" else 30)
                if mutation == "split_strategy":
                    self.config["dataset"]["split_strategy"] = "full_interleaved"
                elif mutation == "train_side":
                    self.config["dataset"]["train_side"] = "after"
                elif mutation in {"source_provenance", "legacy_generator"}:
                    for path in (self.train_path, self.test_path):
                        with h5py.File(path, "r+") as handle:
                            value = json.loads(handle.attrs["fusion_input_signature"])
                            if mutation == "source_provenance":
                                value["source_contract"]["effective_config"]["dataset"]["train_side"] = "after"
                            else:
                                value["source_contract"]["generator_version"] = "moe-learn-python-features-v2"
                            handle.attrs["fusion_input_signature"] = json.dumps(value)
                with self.assertRaises(ValueError):
                    load_prepared_data(self.config, self.train_path, self.test_path)
        self._before_after_fixture(before_count=30, after_count=30, stages=2)
        result = load_prepared_data(self.config, self.train_path, self.test_path)
        self.assertEqual(len(result["stages"]), 2)
        self.assertEqual(result["metadata"]["stages"][1]["train_source_ids"], result["metadata"]["allocation"]["after"]["train_source_ids"])

    def _range_group_fixture(self):
        self._before_after_fixture(before_count=30, after_count=90)
        allocation = load_prepared_data(self.config, self.train_path, self.test_path)["metadata"]["allocation"]["after"]
        # Repeated distances and every edge occur in all three independently fixed roles.
        values = (1., 1.25, 1.75, 2., 2.25, 2.75, 3., 3.25, 3.75, 4.)
        labels = {source_id: values[index % len(values)] for role in ("train", "val", "test")
                  for index, source_id in enumerate(allocation[f"{role}_source_ids"])}
        self._write_after_labels(labels)
        self.config["dataset"].update(after_grouping="range_bins", after_range_edges_km=[1., 2., 3., 4.])
        return labels, allocation

    def _write_after_labels(self, labels):
        self._write_pool_labels(self.test_path, labels)

    def _write_pool_labels(self, path, labels):
        with h5py.File(path, "r+") as handle:
            ids = np.asarray(handle["split/source_segment_idx"]).reshape(-1).astype(int)
            handle["y_range_km"][:] = np.asarray([labels[int(source_id)] for source_id in ids], dtype=np.float64).reshape(-1, 1)

    def test_distance_groups_are_disjoint_and_respect_edges_and_duplicate_distances(self):
        labels, _ = self._range_group_fixture()
        result = load_prepared_data(self.config, self.train_path, self.test_path)
        groups = result["metadata"]["allocation"]["range_groups"]
        self.assertEqual(len(groups), 3)
        source_to_group, distance_to_groups, observed_positions = {}, {}, []
        for number, group in enumerate(groups, 1):
            self.assertEqual(group["group"], number)
            self.assertEqual(group["lower_km"], float(number))
            self.assertEqual(group["upper_km"], float(number + 1))
            self.assertEqual(group["upper_inclusive"], number == 3)
            stage = result["metadata"]["stages"][number]
            for role in ("train", "val", "test"):
                ids = group[f"{role}_source_ids"]
                self.assertGreater(len(ids), 0)
                self.assertEqual(group[f"{role}_count"], len(ids))
                if role != "test":
                    self.assertEqual(stage[f"{role}_source_ids"], ids)
                    self.assertEqual(len(result["stages"][number][f"{role}_y"]), len(ids))
                for source_id in ids:
                    self.assertNotIn(source_id, source_to_group)
                    source_to_group[source_id] = number
                    distance = labels[source_id]
                    self.assertGreaterEqual(distance, group["lower_km"])
                    if group["upper_inclusive"]:
                        self.assertLessEqual(distance, group["upper_km"])
                    else:
                        self.assertLess(distance, group["upper_km"])
                    distance_to_groups.setdefault(distance, set()).add(number)
            positions = group["test_positions"]
            self.assertEqual(group["test_source_ids"], [result["metadata"]["test_source_ids"][i] for i in positions])
            observed_positions.extend(positions)
        self.assertEqual(set(source_to_group), set(labels))
        self.assertEqual(sorted(observed_positions), list(range(len(result["test_y"]))))
        self.assertTrue(all(len(groups_for_distance) == 1 for groups_for_distance in distance_to_groups.values()))
        self.assertEqual({edge: distance_to_groups[edge] for edge in (1., 2., 3., 4.)},
                         {1.: {1}, 2.: {2}, 3.: {3}, 4.: {3}})

    def test_distance_grouping_preserves_all_role_identities_and_stage_one_preprocessing(self):
        self._range_group_fixture()
        range_result = load_prepared_data(self.config, self.train_path, self.test_path)
        repeated = load_prepared_data(self.config, self.train_path, self.test_path)
        self.assertEqual(range_result["metadata"], repeated["metadata"])
        time_config = deepcopy(self.config)
        time_config["dataset"]["after_grouping"] = "time_interleaved"
        del time_config["dataset"]["after_range_edges_km"]
        time_result = load_prepared_data(time_config, self.train_path, self.test_path)
        for key in ("preprocessing", "input_signature", "test_idx", "test_source_ids"):
            self.assertEqual(range_result["metadata"][key], time_result["metadata"][key])
        self.assertEqual(range_result["metadata"]["stages"][0], time_result["metadata"]["stages"][0])
        np.testing.assert_array_equal(range_result["test_x"], time_result["test_x"])
        np.testing.assert_array_equal(range_result["test_y"], time_result["test_y"])
        for role in ("train", "val"):
            range_rows, time_rows = {}, {}
            for result, rows in ((range_result, range_rows), (time_result, time_rows)):
                for index, identity in enumerate(result["metadata"]["stages"]):
                    for position, source_id in enumerate(identity[f"{role}_source_ids"]):
                        self.assertNotIn(source_id, rows)
                        rows[source_id] = (result["stages"][index][f"{role}_x"][position],
                                           result["stages"][index][f"{role}_y"][position])
            self.assertEqual(set(range_rows), set(time_rows))
            for source_id in range_rows:
                np.testing.assert_array_equal(range_rows[source_id][0], time_rows[source_id][0])
                np.testing.assert_array_equal(range_rows[source_id][1], time_rows[source_id][1])

    def test_distance_groups_reject_uncovered_observations(self):
        for edges in ([1.1, 2., 3., 4.], [1., 2., 3., 3.9]):
            with self.subTest(edges=edges):
                self._range_group_fixture()
                self.config["dataset"]["after_range_edges_km"] = edges
                with self.assertRaises(ValueError):
                    load_prepared_data(self.config, self.train_path, self.test_path)

    def test_distance_groups_reject_each_empty_role_even_if_other_roles_fill_that_bin(self):
        for empty_role in ("train", "val", "test"):
            with self.subTest(role=empty_role):
                labels, allocation = self._range_group_fixture()
                for source_id in allocation[f"{empty_role}_source_ids"]:
                    if 2. <= labels[source_id] < 3.:
                        labels[source_id] = 1.5
                self._write_after_labels(labels)
                with self.assertRaises(ValueError):
                    load_prepared_data(self.config, self.train_path, self.test_path)

    def _distance_holdout_fixture(self, heldout_group=2):
        self._before_after_fixture(before_count=60, after_count=90, stages=3)
        for key in ("after_test_fraction", "after_grouping", "after_range_edges_km"):
            self.config["dataset"].pop(key, None)
        self.config["dataset"].update(stage_mode="distance_holdout", distance_edges_km=[1., 2., 3., 4.],
                                      heldout_group=heldout_group)
        values = (1., 1.25, 1.75, 2., 2.25, 2.75, 3., 3.25, 3.75, 4.)
        before = {source: values[(source - 1) % len(values)] for source in range(1, 61)}
        after = {source: values[(source - 1001) % len(values)] for source in range(1001, 1091)}
        self._write_pool_labels(self.train_path, before)
        self._write_pool_labels(self.test_path, after)
        return before, after

    @staticmethod
    def _distance_group(distance):
        # These expected intervals deliberately use no production grouping helper.
        if 1. <= distance < 2.:
            return 1
        if 2. <= distance < 3.:
            return 2
        if 3. <= distance <= 4.:
            return 3
        raise AssertionError(f"Unexpected fixture distance: {distance}")

    def test_all_distance_holdout_folds_exclude_test_ranges_from_every_learning_role(self):
        all_fold_test_ids, all_fold_excluded_ids = [], []
        for heldout in (1, 2, 3):
            with self.subTest(heldout=heldout):
                before, after = self._distance_holdout_fixture(heldout)
                labels = {**before, **after}
                result = load_prepared_data(self.config, self.train_path, self.test_path)
                repeated = load_prepared_data(self.config, self.train_path, self.test_path)
                metadata, allocation = result["metadata"], result["metadata"]["allocation"]
                self.assertEqual(metadata, repeated["metadata"])
                self.assertEqual(allocation["heldout_group"], heldout)
                self.assertEqual(allocation["distance_edges_km"], [1., 2., 3., 4.])
                self.assertEqual(allocation["test_range_excluded_from_learning"], "PASS")
                self.assertEqual(len(result["stages"]), 3)
                expected_test = {source for source, distance in after.items() if self._distance_group(distance) == heldout}
                expected_excluded = {source for source, distance in before.items() if self._distance_group(distance) == heldout}
                test_ids = metadata["test_source_ids"]
                excluded_ids = allocation["before"]["excluded_source_ids"]
                self.assertEqual(set(test_ids), expected_test)
                self.assertEqual(set(excluded_ids), expected_excluded)
                self.assertEqual(allocation["after"]["test_source_ids"], test_ids)
                self.assertEqual(len(result["test_y"]), len(expected_test))
                np.testing.assert_allclose(result["test_y"].reshape(-1), [after[source] for source in test_ids])
                seen = list(test_ids) + list(excluded_ids)
                observed_after_groups = []
                for index, stage in enumerate(metadata["stages"]):
                    self.assertEqual(stage["source_side"], "before" if index == 0 else "after")
                    if index:
                        observed_after_groups.append(stage["range_group"]["group"])
                    for role in ("train", "val"):
                        ids, distances = stage[f"{role}_source_ids"], stage[f"{role}_ranges_km"]
                        self.assertTrue(ids)
                        self.assertEqual(distances, [labels[source] for source in ids])
                        self.assertTrue(all(self._distance_group(distance) != heldout for distance in distances))
                        if index:
                            self.assertEqual({self._distance_group(distance) for distance in distances},
                                             {stage["range_group"]["group"]})
                        np.testing.assert_allclose(result["stages"][index][f"{role}_y"].reshape(-1), distances)
                        seen.extend(ids)
                self.assertEqual(observed_after_groups, [group for group in (1, 2, 3) if group != heldout])
                self.assertEqual(len(seen), len(set(seen)))
                self.assertEqual(set(seen), set(labels))
                for group in allocation["distance_groups"]:
                    self.assertEqual(group["test_count"], len(expected_test) if group["group"] == heldout else 0)
                all_fold_test_ids.extend(test_ids)
                all_fold_excluded_ids.extend(excluded_ids)
        self.assertEqual(sorted(all_fold_test_ids), list(range(1001, 1091)))
        self.assertEqual(sorted(all_fold_excluded_ids), list(range(1, 61)))

    def test_distance_holdout_edges_and_duplicate_distances_have_one_group(self):
        for heldout in (1, 2, 3):
            with self.subTest(heldout=heldout):
                before, after = self._distance_holdout_fixture(heldout)
                result = load_prepared_data(self.config, self.train_path, self.test_path)
                groups = result["metadata"]["allocation"]["distance_groups"]
                self.assertEqual([group["group"] for group in groups], [1, 2, 3])
                for number, group in enumerate(groups, 1):
                    self.assertEqual((group["lower_km"], group["upper_km"], group["upper_inclusive"]),
                                     (float(number), float(number + 1), number == 3))
                test_ids = set(result["metadata"]["test_source_ids"])
                excluded_ids = set(result["metadata"]["allocation"]["before"]["excluded_source_ids"])
                for edge, expected_group in ((1., 1), (2., 2), (3., 3), (4., 3)):
                    edge_test = {source for source, distance in after.items() if distance == edge}
                    edge_excluded = {source for source, distance in before.items() if distance == edge}
                    self.assertGreater(len(edge_test), 1)
                    self.assertGreater(len(edge_excluded), 1)
                    self.assertEqual(test_ids & edge_test, edge_test if heldout == expected_group else set())
                    self.assertEqual(excluded_ids & edge_excluded, edge_excluded if heldout == expected_group else set())

    def test_distance_holdout_preprocessing_uses_only_filtered_before_training_rows(self):
        self._distance_holdout_fixture()
        reference = load_prepared_data(self.config, self.train_path, self.test_path)
        stage = reference["metadata"]["stages"][0]
        train_ids = stage["train_source_ids"]
        with h5py.File(self.train_path, "r+") as handle:
            for index in set(range(60)) - set(stage["train_idx"]):
                handle["X"][index] = 1e6
        with h5py.File(self.test_path, "r+") as handle:
            handle["X"][:] = 1e8
        changed = load_prepared_data(self.config, self.train_path, self.test_path)
        state = changed["metadata"]["preprocessing"]
        self.assertEqual(state, reference["metadata"]["preprocessing"])
        self.assertEqual(state["fit_sample_count"], len(train_ids))
        self.assertEqual(state["fit_source_identity_sha256"],
                         hashlib.sha256(",".join(str(source) for source in train_ids).encode()).hexdigest())
        expected = math.sqrt(2. * sum(source ** 2 for source in train_ids) / len(train_ids))
        self.assertAlmostEqual(state["scales"][0], expected)
        self.assertAlmostEqual(state["scales"][1], expected * 10)
        np.testing.assert_array_equal(reference["stages"][0]["train_x"], changed["stages"][0]["train_x"])

    def test_distance_holdout_rejects_uncovered_before_or_after_distances(self):
        for pool in ("before", "after"):
            for outside in (.9, 4.1):
                with self.subTest(pool=pool, outside=outside):
                    before, after = self._distance_holdout_fixture()
                    labels, path = (before, self.train_path) if pool == "before" else (after, self.test_path)
                    labels[next(iter(labels))] = outside
                    self._write_pool_labels(path, labels)
                    with self.assertRaises(ValueError):
                        load_prepared_data(self.config, self.train_path, self.test_path)

    def test_distance_holdout_rejects_empty_train_validation_test_and_before_learning_pools(self):
        for empty_role in ("train", "val", "test", "before_learning"):
            with self.subTest(empty_role=empty_role):
                before, after = self._distance_holdout_fixture(heldout_group=3)
                reference = load_prepared_data(self.config, self.train_path, self.test_path)
                if empty_role == "before_learning":
                    self._write_pool_labels(self.train_path, {source: 3.5 for source in before})
                else:
                    if empty_role == "test":
                        ids = reference["metadata"]["test_source_ids"]
                    else:
                        ids = reference["metadata"]["stages"][1][f"{empty_role}_source_ids"]
                    for source in ids:
                        after[source] = 2.5
                    self._write_pool_labels(self.test_path, after)
                with self.assertRaises(ValueError):
                    load_prepared_data(self.config, self.train_path, self.test_path)


if __name__ == "__main__":
    unittest.main()
