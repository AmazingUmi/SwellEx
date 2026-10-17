"""Public run.sh study operations: durable remote launch, live status and fetch."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import uuid

from scripts.MoE_data_scale.localization_networks.workflow import remote, study
from scripts.MoE_data_scale.localization_networks.workflow.study_worker import identity, safe_name
from scripts.run_identity import parse_run_id

WORKER = 'scripts/MoE_data_scale/localization_networks/workflow/study_worker.py'


def is_study_id(value):
    if isinstance(value, str) and re.match(r'^\d{8}T\d{6}Z-', value):
        parse_run_id(value, remote.PACK)  # Malformed and foreign child IDs remain errors.
        return False
    safe_name(value)
    return True


RPC = '''import json,pathlib,subprocess,sys
p=json.load(sys.stdin);root=pathlib.Path(p['workspace'])/'studies'/p['study_id']
if root.resolve()!=root:raise ValueError('External study directory')
if p['action']=='install':
 project=pathlib.Path(p['stage'])/'project'
else:
 control=json.loads((root/'control.json').read_text())
 if control.get('pack')!='MoE_data_scale' or control.get('study_id')!=p['study_id']:raise ValueError('Registered study identity mismatch')
 project=pathlib.Path(control['controller_project'])
 if not project.resolve().is_relative_to(root/'controllers'):raise ValueError('External controller snapshot')
command=[sys.executable,'-B',str(project/p['worker']),'--rpc',p['action']]
r=subprocess.run(command,input=json.dumps(p),text=True,capture_output=True)
if r.returncode:raise RuntimeError(r.stderr.strip() or r.stdout.strip())
print(r.stdout)
'''


def rpc(config, study_id, action, **payload):
    safe_name(study_id)
    return remote.remote_python(config, RPC, {'workspace': config['remote']['workspace'],
        'study_id': study_id, 'worker': WORKER, 'action': action, **payload})


def status(config, study_id):
    result = rpc(config, study_id, 'status')
    if result.get('pack') != remote.PACK or result.get('study_id') != study_id:
        raise ValueError('Study status identity mismatch')
    return result


def summarize(report):
    print(f"{report['study_id']}: {report['status']}")
    print(f"  Models: {report['completed_models']}/{report['total_models']} | "
          f"Children: {report['completed_children']}/{report['total_children']} | "
          f"Remote coordinator: {'alive' if report['controller_alive'] else 'not running'}")
    for child in report['children']:
        if child['status'] not in {'PASS', 'PLANNED'}:
            print(f"  {child['status']} {child['run_id']}")
    if report.get('error'):
        print(f"  {report['error']}")


def fetch(config, study_id):
    snapshot = rpc(config, study_id, 'snapshot')
    ledger = snapshot['ledger']
    local = remote.LOCAL_RUNS / safe_name(study_id)
    study.validate_ledger(ledger, study.validate_plan(ledger['plan']), local)
    if local.is_symlink():
        raise ValueError('External local study directory')
    if local.is_symlink() or ((local.exists() and not (local / "study.json").is_file()) and any(local.iterdir())):
        raise ValueError("Existing study directory has no frozen ledger")
    local.mkdir(parents=True, exist_ok=True)
    with (local / 'study.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (local / 'study.json').exists() and identity(study.read_json(local / 'study.json')) != identity(ledger):
            raise ValueError('Fetched study differs from local immutable identity')
        if ledger['plan']['base_config']['remote'] != config['remote']:
            raise ValueError('Remote endpoint differs from study plan')
        remote.dump(local / 'study.json', ledger)
        remote.dump(local / 'remote-control.json', snapshot['control'])
        remote.dump(local / 'remote-status.json', snapshot['report'])
        source = f"{config['remote']['host']}:{config['remote']['workspace']}/studies/{study_id}/coordinator.log"
        subprocess.run(['rsync', '-rlt', '--checksum', '-e', shlex.join(['ssh', *remote.SSH_OPTIONS]),
                        source, str(local / 'coordinator.log')], check=True)
        for child in snapshot['report']['children']:
            if child['status'] != 'PLANNED':
                remote.fetch(config, child['run_id'])
    from scripts.MoE_data_scale.visualization.study_results import summarize_study
    summary = summarize_study(local)
    if summary.get('validation_errors'):
        raise ValueError('Study delivery validation failed: ' + '; '.join(summary['validation_errors']))
    if snapshot['report']['status'] == 'PASS' and summary.get('status') != 'PASS':
        raise ValueError('Completed study delivery is incomplete; inspect study-report.json')
    print(f'Study results (retrieved snapshot): {local}')
    # Never advertise newer PASS/progress whose children were absent from this fetch.
    return snapshot['report']


def run_study(plan_path, *, study_dir=None, dry_run=False):
    plan = study.validate_plan(study.read_json(Path(plan_path)))
    study_id = plan['study_id']
    safe_name(study_id)
    local = remote.LOCAL_RUNS.resolve() / study_id
    if study_dir is not None and (Path(study_dir).is_symlink() or Path(study_dir).resolve() != local):
        raise ValueError('--study-dir must name a direct output/data directory')
    if dry_run:
        cells = [{'seed': seed, 'train_fraction': fraction, 'experts': experts,
                  'network': f'complex_MoE_e{experts}', 'method': plan['method']}
                 for seed, fraction, experts in study.matrix(plan)]
        print(json.dumps({'status': 'NOT_RUN', 'study_id': study_id,
                          'execution_host': plan['base_config']['remote']['host'],
                          'unique_children': len(cells), 'total_models': len(cells),
                          'plan_sha256': study.digest(plan), 'cells': cells}, indent=2))
        return 0
    if local.is_symlink() or ((local.exists() and not (local / "study.json").is_file()) and any(local.iterdir())):
        raise ValueError("Existing study directory has no frozen ledger")
    local.mkdir(parents=True, exist_ok=True)
    with (local / 'study.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (local / 'study.json').exists():
            ledger = study.read_json(local / 'study.json')
            study.validate_ledger(ledger, plan, local)
        else:
            ledger = study.make_ledger(plan, local)
            study.save(local, ledger)
        config = plan['base_config']
        registered = remote.remote_python(config, '''import json,pathlib,sys
p=json.load(sys.stdin);root=pathlib.Path(p['workspace'])/'studies'/p['study_id']
if root.resolve()!=root:raise ValueError('External study directory')
result=None
if (root/'control.json').is_file():
 result={'ledger':json.loads((root/'study.json').read_text()),'training_manifest':json.loads((root/'training-manifest.json').read_text())}
print(json.dumps(result))''', {'workspace': config['remote']['workspace'], 'study_id': study_id})
        if registered is not None and identity(registered['ledger']) != identity(ledger):
            raise ValueError('Registered remote study differs from local immutable identity')
        suite, selection = study.selection_for(plan, ledger['children'][0]['experts'])
        with tempfile.TemporaryDirectory(prefix='swellex-study-controller-') as temporary:
            stage = Path(temporary) / 'project'
            stage.mkdir()
            controller = remote.freeze_source(stage, config, ledger['children'][0]['run_id'], suite, selection=selection)
            origin = None
            if registered is not None:
                training = registered['training_manifest']
            elif controller['source_sha256'] == ledger['source_sha256']:
                training = controller
            else:
                # Existing study: use its original remote source, never today's edited tree.
                candidates = [c for c in ledger['children'] if 'attempted_at' in c]
                for child in candidates:
                    path = remote.local_paths(child['run_id'], config=config)[0] / 'manifest.json'
                    if path.is_file():
                        candidate = study.read_json(path)
                        if candidate.get('source_sha256') == ledger['source_sha256'] and candidate.get('files') == ledger['source_files']:
                            origin, training = child['run_id'], candidate
                            break
                if origin is None:
                    raise ValueError('Original frozen training source is unavailable; refusing source substitution')
            staging = config['remote']['workspace'] + '/study-staging/' + uuid.uuid4().hex
            remote.remote_python(config, "import json,pathlib,sys;p=json.load(sys.stdin);pathlib.Path(p['path']).mkdir(parents=True,exist_ok=False);print('{}')", {'path': staging + '/project'})
            subprocess.run(['rsync', '-rltp', '-e', shlex.join(['ssh', *remote.SSH_OPTIONS]), str(stage) + '/',
                            config['remote']['host'] + ':' + staging + '/project/'], check=True)
            result = rpc(config, study_id, 'install', stage=staging, ledger=ledger,
                         controller_manifest=controller, training_manifest=training, source_run_id=origin)
            remote.dump(local / 'remote-status.json', result)
    print(f'Run: {study_id}', flush=True)
    summarize(result)
    return 0 if result['status'] != 'FAIL' else 1
