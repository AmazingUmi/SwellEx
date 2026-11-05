"""Verify Worker architecture sidecar export and frozen calibration artifact integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.remote_workflow.data_scale_pack.worker import Worker

PROJECT = Path(__file__).resolve().parents[4]


class WorkerCapacityExportTests(unittest.TestCase):
    def test_worker_exports_frozen_architecture_sidecar_to_data_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "20261015T120000Z-scale-full-SCM-all-12345678"
            arch_dir = run_dir / "project/outputs/networks_results/SCM/scm_complex_cnn_range/train_outputs"
            arch_dir.mkdir(parents=True)
            sidecar = arch_dir / "SCM_scm_complex_cnn_range_architecture_0101_000000.json"
            payload = {"schema_version": 1, "scaling": {"selected_profile": "medium", "parameter_count": 12345}}
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "reports").mkdir(parents=True)
            worker = Worker(run_dir)
            worker.export()
            exported = run_dir / "export/data" / sidecar.relative_to(run_dir)
            self.assertTrue(exported.is_file())
            self.assertEqual(json.loads(exported.read_text(encoding="utf-8")), payload)

    def test_frozen_scm_calibration_artifact_preserves_exact_sha256(self):
        cal = PROJECT / "scripts/remote_workflow/data_scale_pack/scm_capacity_calibration_v2.json"
        self.assertTrue(cal.is_file())
        digest = hashlib.sha256(cal.read_bytes()).hexdigest()
        self.assertEqual(digest, "0c30dafb47f9f28ec3c3edfc346bea3b510846f6a2dc8dce7996099493245f19")


if __name__ == "__main__":
    unittest.main()
