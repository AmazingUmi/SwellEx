"""Public remote-study retrieval uses one coherent snapshot; stdlib fixtures only."""
from __future__ import annotations

import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.data_scale_pack.localization_networks.workflow import remote, study, study_remote


class RemoteStudyClientTests(unittest.TestCase):
    def test_original_generated_budget_study_ids_remain_queryable(self):
        self.assertTrue(study_remote.is_study_id("20261015T154330Z-budget-study-1234abcd"))
        for invalid in ("20261315T154330Z-budget-study-1234abcd",
                        "20261015T154330Z-budget-study-short"):
            with self.subTest(run_id=invalid), self.assertRaises(ValueError):
                study_remote.is_study_id(invalid)

    def test_fetch_returns_only_the_snapshot_whose_children_it_retrieved(self):
        config = remote.load_config(remote.DEFAULT_CONFIG)
        plan = {"schema_version": 1, "base_config": config, "fractions": [.25, .5, 1.0],
                "seeds": [2029], "method": "SCM"}
        inventory = [{"path": "README.md", "size": 1, "sha256": "a" * 64}]
        with tempfile.TemporaryDirectory(prefix="swellex-study-fetch-") as temporary:
            local_runs = Path(temporary) / "output" / "data"
            directory = local_runs / "fixture-study"
            directory.mkdir(parents=True)
            with mock.patch.object(study, "source_inventory",
                                   return_value=(inventory, study.digest(inventory))):
                ledger = study.make_ledger(plan, directory)
            ledger.update(status="RUNNING", updated_at="snapshot-time")
            for child, state in zip(ledger["children"], ("PASS", "RUNNING", "PLANNED"), strict=True):
                child["status"] = state
                if state != "PLANNED":
                    child["attempted_at"] = "original-attempt"
            study.validate_ledger(ledger, plan, directory)

            # A previous local view has the same immutable identity, but lags progress.
            previous = copy.deepcopy(ledger)
            previous["updated_at"] = "previous-time"
            previous["children"][0]["status"] = "RUNNING"
            previous["children"][1]["status"] = "PLANNED"
            previous["children"][1].pop("attempted_at")
            study.validate_ledger(previous, plan, directory)
            remote.dump(directory / "study.json", previous)

            report = {"pack": remote.PACK, "study_id": directory.name, "status": "RUNNING",
                      "completed_children": 1, "total_children": 3,
                      "children": [{"run_id": child["run_id"], "status": child["status"]}
                                   for child in ledger["children"]]}
            snapshot = {"ledger": ledger, "report": report, "calibration": None,
                        "control": {"pack": remote.PACK, "study_id": directory.name}}
            later_report = copy.deepcopy(report)
            later_report.update(status="PASS", completed_children=3)
            for child in later_report["children"]:
                child["status"] = "PASS"

            with mock.patch.object(remote, "LOCAL_RUNS", local_runs), \
                    mock.patch.object(study_remote, "rpc", return_value=snapshot) as rpc, \
                    mock.patch.object(study_remote, "status", return_value=later_report) as newer_status, \
                    mock.patch.object(remote, "fetch", return_value={"status": "PASS"}) as fetch_child, \
                    mock.patch.object(study_remote.subprocess, "run") as transfer, \
                    mock.patch("sys.stdout", io.StringIO()) as output:
                actual = study_remote.fetch(config, directory.name)

            rpc.assert_called_once_with(config, directory.name, "snapshot")
            newer_status.assert_not_called()
            fetch_child.assert_has_calls([
                mock.call(config, ledger["children"][0]["run_id"]),
                mock.call(config, ledger["children"][1]["run_id"]),
            ])
            self.assertEqual(fetch_child.call_count, 2)
            self.assertIs(actual, report)
            self.assertEqual(actual["status"], "RUNNING")
            self.assertEqual(actual["completed_children"], 1)
            self.assertEqual(study.read_json(directory / "study.json"), ledger)
            self.assertEqual(study.read_json(directory / "remote-status.json"), report)
            self.assertEqual(study.read_json(directory / "remote-control.json"), snapshot["control"])
            transfer.assert_called_once()
            self.assertEqual(transfer.call_args.args[0][0], "rsync")
            self.assertEqual(transfer.call_args.args[0][-1], str(directory / "coordinator.log"))
            self.assertTrue(transfer.call_args.kwargs["check"])
            self.assertIn("retrieved snapshot", output.getvalue())


class LocalDeliveryPathsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swellex-grouped-delivery-")
        self.addCleanup(temporary.cleanup)
        self.data = Path(temporary.name) / "data"
        self.figures = Path(temporary.name) / "figure"
        for key, value in (("LOCAL_RUNS", self.data), ("LOCAL_FIGURES", self.figures)):
            patch = mock.patch.object(remote, key, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.run_id = "20261015T154330Z-scale-full-SCM-all-9c34e888"
        self.config = remote.load_config(remote.DEFAULT_CONFIG)

    def owner(self, name="fixture-study"):
        remote.dump(self.data / name / "study.json", {
            "pack": remote.PACK, "study_id": name,
            "plan": {"base_config": self.config}, "children": [{"run_id": self.run_id}],
        })

    def test_standalone_run_retains_its_own_delivery_directory(self):
        self.assertEqual(remote.local_paths(self.run_id),
                         (self.data / self.run_id, self.figures / self.run_id))

    def test_repeated_direct_child_fetch_stays_under_owner(self):
        self.owner()
        expected = (self.data / "fixture-study/children" / self.run_id,
                    self.figures / "fixture-study/children" / self.run_id)
        report = {"pack": remote.PACK, "run_id": self.run_id, "status": "PASS", "suite": "quick"}
        with mock.patch.object(remote, "status", return_value=report), \
                mock.patch.object(remote.subprocess, "run", return_value=mock.Mock(returncode=0)) as transfer, \
                mock.patch("sys.stdout", io.StringIO()):
            for _ in range(2):
                remote.fetch(self.config, self.run_id)
        self.assertEqual(remote.local_paths(self.run_id), expected)
        self.assertTrue((expected[0] / "report.json").is_file())
        self.assertEqual(transfer.call_count, 6)
        self.assertEqual([call.args[0][-1] for call in transfer.call_args_list[:3]],
                         [str(expected[0]) + "/", str(expected[1]) + "/", str(expected[0] / "logs") + "/"])
        self.assertFalse((self.data / self.run_id).exists())
        self.assertFalse((self.figures / self.run_id).exists())

    def test_ambiguous_owners_and_loose_duplicates_are_rejected(self):
        self.owner()
        (self.data / self.run_id).mkdir()
        with self.assertRaisesRegex(ValueError, "Ungrouped"):
            remote.local_paths(self.run_id)
        (self.data / self.run_id).rmdir()
        self.owner("second-study")
        with self.assertRaisesRegex(ValueError, "multiple"):
            remote.local_paths(self.run_id)

    def test_owner_endpoint_and_external_delivery_are_rejected(self):
        self.owner()
        other = copy.deepcopy(self.config)
        other["remote"]["workspace"] += "-other"
        with self.assertRaisesRegex(ValueError, "endpoint"):
            remote.local_paths(self.run_id, config=other)
        (self.data / "fixture-study/children").symlink_to(self.data, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            remote.local_paths(self.run_id)


if __name__ == "__main__":
    unittest.main()
