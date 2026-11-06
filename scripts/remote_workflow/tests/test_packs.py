"""Public command, model-filter and descriptive-ID contracts; stdlib only."""
from __future__ import annotations

import copy
import io
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.remote_workflow.packs import PACKS
from scripts.remote_workflow.run import PROJECT, main
from scripts.remote_workflow.run_identity import make_run_id, parse_run_id, validate_run_identity


class PackRoutingTests(unittest.TestCase):
    def test_presets_and_independent_model_filters(self):
        for pack in PACKS.values():
            remote = import_module(pack.dispatcher)
            for preset, split, side in [('full','full_interleaved','before'),('train_before','range_overlap','before'),('train_after','range_overlap','after')]:
                for method, network in [(None,None), (pack.methods[-1],None), (None,pack.networks[-1]), (pack.methods[-1],'ResNet18' if 'ResNet18' in pack.networks else pack.networks[1])]:
                    args=[pack.name,'--preset',preset]
                    if method: args += ['--method',method]
                    if network: args += ['--network',network]
                    with self.subTest(args=args), patch.object(remote,'dispatch',return_value=0) as dispatch:
                        self.assertEqual(main(args),0)
                    config,suite,no_cache=dispatch.call_args.args
                    self.assertEqual((config['dataset']['split_strategy'],config['dataset']['train_side']),(split,side))
                    self.assertEqual(config['training']['epochs'],100)
                    selection=dispatch.call_args.kwargs.get('selection')
                    plan=remote.model_plan(suite,selection)
                    self.assertEqual(len(plan),(1 if method else len(pack.methods))*(1 if network else len(pack.networks)))
                    self.assertEqual({job['method'] for job in plan}, {method} if method else set(pack.methods))
                    self.assertEqual({job['architecture'] for job in plan}, {network} if network else set(pack.networks))
                    self.assertFalse(no_cache)

    def test_custom_config_override(self):
        for pack in PACKS.values():
            remote=import_module(pack.dispatcher)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'custom config; $(literal).toml'
                path.write_text(remote.DEFAULT_CONFIG.read_text().replace('epochs = 100','epochs = 3'))
                args=[pack.name,'--config',str(path),'--method',pack.methods[-1],'--no-cache']
                with patch.object(remote,'dispatch',return_value=0) as dispatch:
                    self.assertEqual(main(args),0)
                config,suite,no_cache=dispatch.call_args.args
                self.assertEqual(config['training']['epochs'],3)
                self.assertTrue(no_cache)

    def test_old_and_misplaced_parameters_are_rejected_without_launch(self):
        invalid=[['RBD','complex_CNN'],['run'],['--epochs','1'],['--seed','0'],['--detach'],['--dry-run'],['--no-wait'],
                 ['--method','all'],['--network','resnet18'],['--method','ELM'],['--network','real_CNN'],['--study-dir','x'],['--method','RBD','--method','SCM'],
                 ['--method','SCM','SCM'],['--network','GRNN','GRNN'],
                 ['local-grnn','--method','RBD','SCM','--python',sys.executable],
                 ['--preset','train_before','--config','missing.toml'],['--preset','before'],['--suite','quick'],
                 ['validate','--method','SCM'],['status'],['fetch','--run','x','--network','GRNN'],['check','--preset','full']]
        for pack in PACKS.values():
            remote=import_module(pack.dispatcher)
            for args in invalid:
                with self.subTest(pack=pack.name,args=args), patch.object(remote,'dispatch') as dispatch, patch.object(remote,'remote_python') as ssh, patch('sys.stderr',io.StringIO()), self.assertRaises(SystemExit):
                    main([pack.name,*args])
                dispatch.assert_not_called(); ssh.assert_not_called()
        with patch('sys.stderr',io.StringIO()), self.assertRaises(SystemExit): main(['base_pack','--fusion','raw_concat'])

    def test_help_and_missing_pack_never_load_config_or_launch(self):
        for pack in PACKS.values():
            remote=import_module(pack.dispatcher)
            with patch.object(remote,'load_config') as load, patch('sys.stdout',io.StringIO()) as output, self.assertRaises(SystemExit) as result:
                main([pack.name,'--config','missing.toml','--help'])
            self.assertEqual(result.exception.code,0); load.assert_not_called()
            self.assertIn('--network',output.getvalue()); self.assertNotIn('--detach',output.getvalue())
        for args in ([],['RBD','complex_CNN'],['BASE_PACK']):
            with patch('sys.stderr',io.StringIO()), self.assertRaises(SystemExit): main(args)

    def test_validation_and_administration(self):
        for pack in PACKS.values():
            remote=import_module(pack.dispatcher)
            for args,suite in [(['validate'],'smoke'),(['validate','--suite','quick'],'quick'),([], 'experiment')]:
                with patch.object(remote,'dispatch',return_value=0) as dispatch:
                    self.assertEqual(main([pack.name,*args]),0)
                self.assertEqual(dispatch.call_args.args[1],suite)
            run_id=make_run_id(pack.name,remote.load_config(remote.DEFAULT_CONFIG))
            for command in ('status','fetch'):
                with patch.object(remote,command,return_value={'run_id':run_id,'status':'PASS'}) as action, patch('sys.stdout',io.StringIO()):
                    self.assertEqual(main([pack.name,command,'--run',run_id]),0)
                self.assertEqual(action.call_args.args[1],run_id)

    def test_multi_value_cli_and_experiment_table_scope(self):
        for pack in PACKS.values():
            remote = import_module(pack.dispatcher)
            with patch.object(remote, 'dispatch', return_value=0) as dispatch:
                self.assertEqual(
                    main([pack.name, '--method', 'SCM', 'RBD', '--network', 'GRNN', 'complex_CNN']),
                    0,
                )
            cfg, suite, no_cache = dispatch.call_args.args
            selection = dispatch.call_args.kwargs['selection']
            self.assertEqual(suite, 'selected')
            self.assertEqual(selection, {'methods': ['SCM', 'RBD'], 'networks': ['GRNN', 'complex_CNN']})
            plan = remote.model_plan(suite, selection)
            self.assertEqual(
                [(j['method'], j['architecture']) for j in plan],
                [('RBD', 'complex_CNN'), ('RBD', 'GRNN'), ('SCM', 'complex_CNN'), ('SCM', 'GRNN')],
            )
            with tempfile.TemporaryDirectory() as tmp:
                subset_toml = Path(tmp) / 'subset.toml'
                text = remote.DEFAULT_CONFIG.read_text()
                text = text.replace('methods = ["RBD", "SCM"]', 'methods = ["SCM"]')
                text = text.replace(
                    'networks = ["complex_CNN", "ResNet18", "ResNet50", "GRNN"]',
                    'networks = ["complex_CNN", "GRNN"]',
                )
                subset_toml.write_text(text)
                with patch.object(remote, 'dispatch', return_value=0) as dispatch:
                    self.assertEqual(main([pack.name, '--config', str(subset_toml)]), 0)
                _, suite, _ = dispatch.call_args.args
                self.assertEqual(suite, 'selected')
                self.assertEqual(
                    dispatch.call_args.kwargs['selection'],
                    {'methods': ['SCM'], 'networks': ['complex_CNN', 'GRNN']},
                )

    def test_imports_do_not_load_numerical_dependencies(self):
        code="""import importlib.abc,sys
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy','scipy','h5py','matplotlib'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Reject())
from scripts.remote_workflow.run import main
main(['base_pack','--help'])
"""
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=PROJECT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


class RunIdentityTests(unittest.TestCase):
    def test_id_uses_effective_config_filters_and_pack(self):
        for pack in PACKS.values():
            remote=import_module(pack.dispatcher)
            for side in ('before','after'):
                cfg=remote.load_config(remote.DEFAULT_CONFIG)
                cfg['dataset'].update(split_strategy='range_overlap',train_side=side)
                for selection in (None, {'method':pack.methods[-1],'network':None}, {'method':None,'network':pack.networks[-1]}):
                    run=make_run_id(pack.name,cfg,selection)
                    fields=parse_run_id(run,pack.name)
                    self.assertEqual(fields['split'],side)
                    self.assertEqual(fields['method'],(selection or {}).get('method') or 'all')
                    validate_run_identity(run,pack.name,cfg,selection)
                    changed=copy.deepcopy(cfg);changed['dataset']['train_side']='after' if side=='before' else 'before'
                    with self.assertRaises(ValueError): validate_run_identity(run,pack.name,changed,selection)
                    self.assertNotEqual(run,make_run_id(pack.name,cfg,selection))

    def test_multi_network_subset_run_id_encoding_and_canonical_order(self):
        for pack in PACKS.values():
            remote = import_module(pack.dispatcher)
            cfg = remote.load_config(remote.DEFAULT_CONFIG)
            selection = {'methods': ['SCM'], 'networks': ['GRNN', 'complex_CNN']}
            run = make_run_id(pack.name, cfg, selection)
            fields = parse_run_id(run, pack.name)
            self.assertEqual(fields['method'], 'SCM')
            self.assertEqual(fields['network'], 'complex_CNN+GRNN')
            validate_run_identity(run, pack.name, cfg, selection)
            for bad in (
                '20261013T120000Z-base-full-SCM-GRNN+complex_CNN-12345678',
                '20261013T120000Z-base-full-SCM-GRNN+GRNN-12345678',
                '20261013T120000Z-base-full-RBD+SCM-all-12345678',
                '20261013T120000Z-base-full-all-complex_CNN+ResNet18+ResNet50+GRNN-12345678',
            ):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    parse_run_id(bad, pack.name)

    def test_old_unsafe_and_cross_pack_ids_rejected_before_ssh(self):
        for pack in PACKS.values():
            remote=import_module(pack.dispatcher)
            invalid=['20261013T120000Z-12345678','../escape','20261313T120000Z-base-full-all-all-12345678',
                     '20261013T120000Z-base-full-ELM-all-12345678','20261013T120000Z-base-full-RBD_SCM-all-12345678',
                     '20261013T120000Z-base-full-all-real_CNN-12345678','20261013T120000Z-base-full-all-all-block_rms-12345678',
                     '20261013T120000Z-joint-full-all-all-12345678','20261013T120000Z-moe-full-all-all-12345678']
            for run in invalid:
                with self.subTest(run=run),patch.object(remote,'remote_python') as ssh,self.assertRaises(ValueError):remote.status(remote.load_config(remote.DEFAULT_CONFIG),run)
                ssh.assert_not_called()


if __name__=='__main__': unittest.main()
