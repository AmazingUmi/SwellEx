"""Pure stdlib matrix identity and dispatch contracts; no numerical work."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.MoE_data_scale.localization_networks.workflow import remote, study, study_remote
from scripts.MoE_data_scale.localization_networks.workflow.study_worker import identity
from scripts.run import main


class ExpertStudyTests(unittest.TestCase):
    def setUp(self):
        self.plan_path = Path(study.__file__).with_name('rbd_scm_expert_study_v1.json')
        self.plan = study.validate_plan(study.read_json(self.plan_path))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.directory = self.root / self.plan['study_id']
        files = [{'path': 'uv.lock', 'size': 4, 'sha256': 'a' * 64}]
        with patch.object(study, 'source_inventory', return_value=(files, study.digest(files))):
            self.ledger = study.make_ledger(self.plan, self.directory)

    def test_default_has_exact_48_explicit_selected_cells(self):
        self.assertEqual(len(self.ledger['children']), 48)
        self.assertEqual(len({c['run_id'] for c in self.ledger['children']}), 48)
        for child in self.ledger['children']:
            config = study.child_config(self.plan, child['seed'], child['train_fraction'], child['experts'])
            suite, filters = study.selection_for(self.plan, child['experts'])
            self.assertEqual(suite, 'selected')
            self.assertEqual(remote.model_plan(suite, filters), [{'method': 'RBD_SCM',
                'architecture': child['network'], 'model': child['model']}])
            self.assertEqual((config['training']['epochs'], config['training']['batch_size'],
                              config['training']['base_channels']), (100, 64, 16))
        study.validate_ledger(self.ledger, self.plan, self.directory)

    def test_plan_rejects_unknown_duplicate_and_invalid_dimensions(self):
        for key, value in [('experts', [1, 1]), ('experts', [True]), ('experts', [3]),
                           ('fractions', [1, True]), ('fractions', [float('nan')]),
                           ('fractions', [0]), ('seeds', [2026, 2026]),
                           ('seeds', [-1]), ('method', 'SCM'), ('study_id', '../bad')]:
            changed = copy.deepcopy(self.plan)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                study.validate_plan(changed)
        changed = {**self.plan, 'budget': 100}
        with self.assertRaises(ValueError):
            study.validate_plan(changed)
        path = self.root / 'duplicate.json'
        path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, 'Repeated JSON'):
            study.read_json(path)

    def test_ledger_rejects_reordering_model_config_source_and_attempt_tampering(self):
        variants = []
        for key, value in [('experts', 8), ('model', 'scm_complex_moe_e1_range'),
                           ('network', 'complex_MoE'), ('config_sha256', 'b' * 64),
                           ('status', 'PASS'), ('train_fraction', True)]:
            changed = copy.deepcopy(self.ledger)
            changed['children'][0][key] = value
            variants.append(changed)
        changed = copy.deepcopy(self.ledger)
        changed['children'][0], changed['children'][1] = changed['children'][1], changed['children'][0]
        variants.append(changed)
        changed = copy.deepcopy(self.ledger)
        changed['source_files'][0]['path'] = '../escape'
        changed['source_sha256'] = study.digest(changed['source_files'])
        variants.append(changed)
        for changed in variants:
            with self.subTest(changed=changed['children'][0]), self.assertRaises(ValueError):
                study.validate_ledger(changed, self.plan, self.directory)
        progress = copy.deepcopy(self.ledger)
        progress.update(status='RUNNING', updated_at='later')
        progress['children'][0].update(status='RUNNING', attempted_at='now')
        self.assertEqual(identity(progress), identity(self.ledger))
        study.validate_ledger(progress, self.plan, self.directory)

    def test_dry_run_uses_public_entry_and_never_ssh_or_writes(self):
        with patch.object(remote, 'remote_python', side_effect=AssertionError('SSH forbidden')), \
                patch.object(remote, 'dump', side_effect=AssertionError('writes forbidden')), \
                patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir forbidden')), \
                patch('sys.stdout', io.StringIO()) as output:
            self.assertEqual(main(['MoE_data_scale', 'study', '--plan', str(self.plan_path), '--dry-run']), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result['status'], 'NOT_RUN')
        self.assertEqual((result['total_models'], len(result['cells'])), (48, 48))
        self.assertEqual({c['experts'] for c in result['cells']}, {1, 2, 4, 8})

    def test_status_distinguishes_old_moe_children_and_cross_pack_ids(self):
        legacy = '20261016T120000Z-moe-full-SCM-complex_MoE_e1-12345678'
        self.assertFalse(study_remote.is_study_id(legacy))
        self.assertTrue(study_remote.is_study_id(self.plan['study_id']))
        for value in ('../escape', '20261016T120000Z-scale-full-SCM-all-12345678',
                      '20261016T120000Z-moe-full-SCM-GRNN-12345678'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                study_remote.is_study_id(value)

    def test_fetch_rejects_invalid_or_incomplete_pass_delivery_and_preserves_remote_evidence(self):
        from scripts.MoE_data_scale.visualization import study_results
        config = self.plan['base_config']
        for execution, summary, fails in (
                ('PASS', {'status': 'PARTIAL', 'validation_errors': ['wrong source']}, True),
                ('PASS', {'status': 'PARTIAL', 'validation_errors': []}, True),
                ('RUNNING', {'status': 'PARTIAL', 'validation_errors': []}, False),
                ('PASS', {'status': 'PASS', 'validation_errors': []}, False)):
            snapshot = {'ledger': copy.deepcopy(self.ledger), 'control': {},
                        'report': {'status': execution, 'children': []}}
            with self.subTest(execution=execution, summary=summary), \
                    patch.object(remote, 'LOCAL_RUNS', self.root), \
                    patch.object(study_remote, 'rpc', return_value=snapshot), \
                    patch.object(study_remote.subprocess, 'run'), \
                    patch.object(study_results, 'summarize_study', return_value=summary), \
                    patch('sys.stdout', io.StringIO()):
                if fails:
                    with self.assertRaises(ValueError):
                        study_remote.fetch(config, self.plan['study_id'])
                else:
                    self.assertEqual(study_remote.fetch(config, self.plan['study_id'])['status'], execution)
            saved = study.read_json(self.directory / 'remote-status.json')
            self.assertEqual(saved['status'], execution)

    def test_children_are_grouped_and_owning_ledger_is_strictly_verified(self):
        data, figures = self.root / 'data', self.root / 'figures'
        owner = data / self.plan['study_id']
        remote.dump(owner / 'study.json', self.ledger)
        unrelated = data / 'unrelated' / 'study.json'
        unrelated.parent.mkdir()
        unrelated.write_text('not JSON')
        child = self.ledger['children'][0]
        with patch.object(remote, 'LOCAL_RUNS', data), patch.object(remote, 'LOCAL_FIGURES', figures):
            paths = remote.local_paths(child['run_id'], config=self.plan['base_config'])
            self.assertEqual(paths, (owner / 'children' / child['run_id'],
                figures / self.plan['study_id'] / 'children' / child['run_id']))
            altered = copy.deepcopy(self.ledger)
            altered['children'][0]['network'] = 'complex_MoE'
            remote.dump(owner / 'study.json', altered)
            with self.assertRaises(ValueError):
                remote.local_paths(child['run_id'], config=self.plan['base_config'])


if __name__ == '__main__':
    unittest.main()
