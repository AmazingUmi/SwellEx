"""New pack dispatch, provenance and closed configuration contracts."""
from __future__ import annotations
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import redirect_stdout

from scripts.joint_feature_pack.localization_networks.workflow import config, remote, tasks
from scripts.joint_feature_pack.localization_networks.workflow.worker import Worker

WORKFLOW = Path(__file__).resolve().parents[1]
RUN = "20261013T080000Z-joint-full-all-all-block_rms-01234567"


def fixture():
    return config.load_config(WORKFLOW / "haiqin_full_interleaved.toml")


class JointWorkflowTests(unittest.TestCase):
    def test_seven_inputs_five_models_and_explicit_joint_phase(self):
        self.assertEqual(len(tasks.METHODS), 7)
        plan = tasks.model_plan("experiment")
        self.assertEqual(len(plan), 35)
        self.assertEqual(len({(p['method'], p['model']) for p in plan}), 35)
        self.assertEqual(len(tasks.model_plan("smoke")), 14)
        for row in plan:
            self.assertEqual(tasks.validate_selection(row), row)
            self.assertIn("joint_datasets", tasks.expected_phases("selected", {"method": row["method"], "network": row["architecture"]}))
            self.assertTrue(tasks.entry_script(row).startswith("scripts/joint_feature_pack/"))

    def test_only_joint_and_grnn_network_entrypoints_are_executable(self):
        root = WORKFLOW.parent
        expected = {
            "scripts/joint_feature_pack/localization_networks/joint/main.py",
            "scripts/joint_feature_pack/localization_networks/grnn/main.py",
        }
        self.assertEqual({tasks.entry_script(row) for row in tasks.model_plan("experiment")}, expected)
        self.assertEqual({path.relative_to(root).as_posix() for path in root.rglob("main.py")},
                         {"joint/main.py", "grnn/main.py"})
        for method in ("rbd", "elm", "scm"):
            for name in ("cli", "bindings", "training", "prediction"):
                self.assertFalse((root / method / "network" / f"{name}.py").exists())

    def test_closed_config_and_alignment_rejections(self):
        for mutate in (lambda c: c['fusion'].update(mode='weighted'),
                       lambda c: c['fusion'].update(extra=1),
                       lambda c: c['dataset'].update(num_snapshots_per_segment=2),
                       lambda c: c.pop('fusion')):
            value = fixture(); mutate(value)
            with self.assertRaises(ValueError): config.validate_config(value)

    def test_fusion_override_and_grnn_epoch_rejection(self):
        with patch.object(remote, 'dispatch', return_value=0) as dispatch:
            self.assertEqual(remote.main(['--method', 'RBD_ELM_SCM', '--network', 'ResNet18', '--fusion', 'raw_concat']), 0)
            self.assertEqual(dispatch.call_args.args[0]['fusion']['mode'], 'raw_concat')
            self.assertEqual(dispatch.call_args.args[0]['training']['epochs'], 100)
        with patch.object(remote, 'dispatch') as dispatch:
            with self.assertRaises(ValueError):
                remote.selected_run(fixture(), 'RBD_SCM', 'invalid', False)
            dispatch.assert_not_called()

    def test_pack_mismatch_rejected_before_fetch(self):
        for pack in (None, 'base_pack'):
            with patch.object(remote, 'remote_python', return_value={'pack': pack, 'run_id': RUN, 'status': 'PASS'}):
                with self.assertRaises(ValueError): remote.status(fixture(), RUN)
                with patch.object(remote.subprocess, 'run') as subprocess:
                    with self.assertRaises(ValueError): remote.fetch(fixture(), RUN)
                    subprocess.assert_not_called()

    def test_manifest_pack_and_source_inventory(self):
        paths = [Path('scripts/joint_feature_pack/localization_networks/joint/main.py'),
                 Path('scripts/base_pack/localization_networks/workflow/worker.py')]
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(remote, 'source_paths', return_value=paths), patch.object(remote.subprocess, 'run', return_value=SimpleNamespace(stdout='test-snapshot')):
                manifest = remote.freeze_source(Path(temporary), fixture(), RUN, 'quick')
            for item in manifest['files']:
                self.assertEqual(remote.sha256(Path(temporary)/item['path']), item['sha256'])
        self.assertEqual(manifest['pack'], 'joint_feature_pack')
        self.assertEqual({r['path'] for r in manifest['files']}, {p.as_posix() for p in paths})

    def test_worker_pack_mismatch_fails_before_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/RUN; root.mkdir()
            request={'schema_version':1, 'pack':'base_pack', 'run_id':RUN, 'suite':'quick', 'config':fixture(), 'use_cache':False}
            (root/'request.json').write_text(json.dumps(request))
            with self.assertRaisesRegex(ValueError, 'pack mismatch'): Worker(root).setup()


if __name__ == '__main__': unittest.main()
