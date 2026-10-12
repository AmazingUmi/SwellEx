"""Remote small-fixture checks for local GRNN source and reused-input evidence."""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from scripts.localization_networks.grnn.tests.test_range_overlap import (
    write_numeric_inputs, write_small_config,
)
from scripts.localization_networks.workflow import compare_results, local_grnn


class LocalGrnnProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swellex local provenance ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        write_numeric_inputs(self.root)
        self.config = write_small_config(self.root, "before")

    def run_grnn(self, data=None):
        with redirect_stdout(io.StringIO()):
            return local_grnn.run(self.config, method="ELM", data=data, root=self.root)

    def test_inventory_keeps_dynamic_loaders_and_init_without_git_or_caches(self):
        project = self.root / "source tree"
        wanted = ["scripts/__init__.py", "scripts/localization_networks/__init__.py"]
        for method in ("rbd", "elm", "scm"):
            wanted.extend((f"scripts/localization_networks/{method}/__init__.py",
                           f"scripts/localization_networks/{method}/network/__init__.py",
                           f"scripts/localization_networks/{method}/network/data.py"))
        ignored = ["scripts/__pycache__/stale.py", "scripts/._junk.py",
                   "scripts/.pytest_cache/generated.py"]
        for name in wanted + ignored:
            path = project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# isolated source fixture\n")
        self.assertEqual([path.relative_to(project).as_posix()
                          for path in local_grnn.python_source_paths(project)], sorted(wanted))
        self.assertFalse((project / ".git").exists())

    def test_generated_and_reused_pass_reports_merge_with_the_same_real_raw_identity(self):
        original = self.run_grnn()
        directory = Path(original["dataset"]["directory"])
        metadata = json.loads(Path(original["dataset"]["metadata_file"]).read_text())
        loaded = local_grnn.load_inputs(self.root, metadata["effective_config"])
        self.assertEqual(metadata["input_files"], loaded["input_files"])
        expected_raw = [local_grnn.file_record(Path(item["path"])) for item in loaded["input_files"]]
        self.assertEqual(original["raw_inputs"], expected_raw)
        reused = self.run_grnn(directory)
        self.assertEqual(reused["status"], "PASS")
        self.assertEqual(reused["raw_inputs"], expected_raw)
        self.assertNotIn("generate_elm", [check["phase"] for check in reused["checks"]])
        source_paths = {Path(item["path"]).relative_to(local_grnn.ROOT).as_posix()
                        for item in reused["source_files"]}
        self.assertIn("scripts/localization_networks/elm/network/data.py", source_paths)
        self.assertIn("scripts/localization_networks/elm/network/__init__.py", source_paths)
        original_records = compare_results.run_records(self.root / "output/data" / original["run_id"], {})
        reused_records = compare_results.run_records(self.root / "output/data" / reused["run_id"], {})
        self.assertEqual(len(reused_records), 1)
        self.assertEqual(reused_records[0]["method"], "ELM")
        self.assertEqual(reused_records[0]["raw_input_identity_sha256"],
                         original_records[0]["raw_input_identity_sha256"])
        self.assertEqual(reused_records[0]["source_verification"], "saved_snapshot_hashes_verified")

    def test_reuse_rejects_missing_inventory_and_changed_raw_bytes_even_with_same_size_and_mtime(self):
        original = self.run_grnn()
        directory = Path(original["dataset"]["directory"])
        path = Path(original["dataset"]["metadata_file"])
        saved_text = path.read_text()
        metadata = json.loads(saved_text)
        bad_hash = copy.deepcopy(metadata["input_files"])
        bad_hash[0]["sha256"] = "0" * 64
        duplicate = copy.deepcopy(metadata["input_files"])
        duplicate[-1] = duplicate[0]
        for inventory in (None, [], metadata["input_files"][:-1], bad_hash, duplicate):
            changed = copy.deepcopy(metadata)
            if inventory is None:
                changed.pop("input_files")
            else:
                changed["input_files"] = inventory
            path.write_text(json.dumps(changed))
            with self.subTest(inventory=inventory), self.assertRaisesRegex(ValueError, "raw input"):
                self.run_grnn(directory)
        path.write_text(saved_text)
        raw_path = Path(metadata["input_files"][0]["path"])
        before = raw_path.read_bytes()
        stat = raw_path.stat()
        changed = bytearray(before)
        changed[0] ^= 1
        raw_path.write_bytes(changed)
        os.utime(raw_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(raw_path.stat().st_size, stat.st_size)
        self.assertEqual(raw_path.stat().st_mtime_ns, stat.st_mtime_ns)
        with self.assertRaisesRegex(ValueError, "raw input identity differs"):
            self.run_grnn(directory)

    def test_selected_loader_change_during_reuse_is_rejected_by_provenance_check(self):
        original = self.run_grnn()
        source_project = self.root / "isolated source"
        shutil.copytree(local_grnn.ROOT / "scripts", source_project / "scripts",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"))
        for name in ("pyproject.toml", ".python-version", "uv.lock"):
            shutil.copyfile(local_grnn.ROOT / name, source_project / name)
        loader = source_project / "scripts/localization_networks/elm/network/data.py"
        predict = local_grnn.predict_command

        def predict_then_change_loader(args):
            predict(args)
            loader.write_text(loader.read_text() + "\n# changed after prediction\n")

        previous = set((self.root / "output/data").iterdir())
        with mock.patch.object(local_grnn, "ROOT", source_project), \
             mock.patch.object(local_grnn, "predict_command", side_effect=predict_then_change_loader), \
             self.assertRaisesRegex(ValueError, "source changed.*elm/network/data.py"):
            self.run_grnn(Path(original["dataset"]["directory"]))
        created = set((self.root / "output/data").iterdir()) - previous
        self.assertEqual(len(created), 1)
        report = json.loads((created.pop() / "report.json").read_text())
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["checks"][-1]["phase"], "verify_provenance")
        self.assertEqual(report["checks"][-1]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
