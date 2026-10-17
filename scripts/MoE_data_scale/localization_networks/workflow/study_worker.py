"""Remote-only durable study controller, separate from frozen training source."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[4]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts.MoE_data_scale.localization_networks.workflow import remote, study
from scripts.MoE_data_scale.localization_networks.workflow.worker import verify_snapshot
from scripts.run_identity import parse_run_id

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
TERMINAL = {"PASS", "FAIL"}


def safe_name(value):
    if not isinstance(value, str) or not NAME.fullmatch(value) or value in {'.', '..'}:
        raise ValueError('Invalid study ID')
    # Never interpret a malformed or foreign child ID as a study.
    if re.match(r'^\d{8}T\d{6}Z-(?:base|joint|scale|moe)-', value):
        raise ValueError('Child run ID is not a study ID')
    return value


def root_for(workspace, study_id):
    workspace = Path(workspace)
    if not workspace.is_absolute() or workspace.resolve() != workspace:
        raise ValueError('Workspace must be an absolute non-symlink path')
    root = workspace / 'studies' / safe_name(study_id)
    if root.resolve() != root:
        raise ValueError('External study directory')
    return root


def identity(ledger):
    result = {k: v for k, v in ledger.items() if k not in {'status', 'error', 'updated_at', 'children'}}
    result['children'] = [{k: v for k, v in child.items() if k not in
                          {'status', 'error', 'attempted_at'}}
                         for child in ledger['children']]
    return study.digest(result)


def process_identity(pid):
    # /proc comm may contain spaces; fields after its closing parenthesis are stable.
    parts = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'start_ticks': parts[19], 'state': parts[0]}


def alive(marker):
    try:
        saved = study.read_json(marker)
        current = process_identity(saved['pid'])
        return (current['state'] != 'Z' and all(current[k] == saved[k]
                                               for k in ('pid', 'boot_id', 'start_ticks')))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def verify_training_source(root, ledger):
    manifest = study.read_json(root / 'training-manifest.json')
    if (manifest.get('pack') != remote.PACK or manifest.get('source_sha256') != ledger['source_sha256']
            or manifest.get('files') != ledger['source_files']):
        raise ValueError('Training snapshot differs from original study source')
    verify_snapshot(root / 'training-source', manifest)
    return manifest


def await_launch(process, marker):
    for _ in range(100):
        if marker.is_file() and study.read_json(marker).get('pid') == process.pid:
            return
        if process.poll() is not None:
            raise RuntimeError('Background process exited before recording its launch identity')
        time.sleep(.05)
    raise RuntimeError('Background launch is ambiguous; inspect status before recovery')


def copy_snapshot(source, target, manifest):
    """Copy only allowlisted source, never outputs, checkpoints or raw inputs."""
    target.mkdir(parents=True, exist_ok=False)
    for item in manifest['files']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe frozen source path')
        origin = source / rel
        if (origin.is_symlink() or not origin.is_file() or not origin.resolve().is_relative_to(source.resolve())
                or origin.stat().st_size != item['size'] or remote.sha256(origin) != item['sha256']):
            raise ValueError(f'Frozen source differs: {rel}')
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)
    verify_snapshot(target, manifest)


def validate(ledger, root):
    plan = study.validate_plan(ledger['plan'])
    study.validate_ledger(ledger, plan, root)


def child_config(ledger, child):
    return study.child_config(ledger['plan'], child['seed'], child['train_fraction'], child['experts'])


def suite_selection(ledger, child):
    return study.selection_for(ledger['plan'], child['experts'])


def child_status(workspace, run_id):
    parse_run_id(run_id, remote.PACK)
    run = Path(workspace) / 'runs' / run_id
    if not run.exists():
        return None
    if not (run / 'report.json').is_file():
        return {'pack': remote.PACK, 'run_id': run_id, 'status': 'UNKNOWN',
                'error': 'Child initialization has no report; refusing automatic relaunch'}
    report = study.read_json(run / 'report.json')
    if report.get('pack') != remote.PACK or report.get('run_id') != run_id:
        raise ValueError('Child report identity mismatch')
    if report['status'] not in TERMINAL and not alive(run / 'launch.json'):
        report = {**report, 'status': 'UNKNOWN', 'error': 'Child has no live launch identity; refusing automatic relaunch'}
    return report


def verify_completed(ledger, child):
    """Use exact request/source/model checks against remote exports."""
    workspace = Path(ledger['plan']['base_config']['remote']['workspace'])
    original_paths, original_verify = remote.local_paths, remote.verify_visualization_delivery
    try:
        remote.local_paths = lambda run_id, **kwargs: (workspace / 'runs' / run_id,
                                                       workspace / 'runs' / run_id / 'export/figure')
        remote.verify_visualization_delivery = lambda local, figures, run_id: original_verify(
            local / 'export/data', local / 'export/figure', run_id)
        study.verify_child(ledger, child, child_config(ledger, child))
    finally:
        remote.local_paths, remote.verify_visualization_delivery = original_paths, original_verify


def inspect(root):
    ledger = study.read_json(root / 'study.json')
    validate(ledger, root)
    control = study.read_json(root / 'control.json')
    if control['pack'] != remote.PACK or control['study_identity'] != identity(ledger):
        raise ValueError('Registered study identity differs')
    workspace = ledger['plan']['base_config']['remote']['workspace']
    running = alive(root / 'controller-launch.json')
    children = []
    complete_models = 0
    for child in ledger['children']:
        suite, selection = suite_selection(ledger, child)
        model_count = len(remote.model_plan(suite, selection))
        report = child_status(workspace, child['run_id'])
        state = report['status'] if report else ('UNKNOWN' if 'attempted_at' in child else 'PLANNED')
        if state == 'PASS':
            verify_completed(ledger, child)
        if running and child['status'] in {'PLANNED', 'DISPATCHING'} and state == 'UNKNOWN':
            state = 'DISPATCHING'
        evaluations = [c for c in (report or {}).get('checks', [])
                       if c.get('status') == 'PASS' and c.get('name', '').startswith('evaluate_')]
        count = model_count if state == 'PASS' else len(evaluations)
        complete_models += count
        children.append({**child, 'status': state, 'completed_models': count})
    state = ledger['status']
    if state not in TERMINAL:
        state = 'RUNNING' if running else 'INTERRUPTED'
    if state == 'PASS' and any(c['status'] != 'PASS' for c in children):
        state = 'FAIL'
    return {'pack': remote.PACK, 'kind': 'remote-study', 'run_id': ledger['study_id'],
            'study_id': ledger['study_id'], 'status': state, 'controller_alive': running,
            'updated_at': ledger['updated_at'], 'children': children,
            'completed_children': sum(c['status'] == 'PASS' for c in children),
            'total_children': len(children), 'completed_models': complete_models,
            'total_models': len(children),
            'source_sha256': ledger['source_sha256'], 'controller_source_sha256': control['controller_source_sha256'],
            'error': ledger.get('error') or (None if state != 'INTERRUPTED' else
                     'Remote coordinator is not alive; repeat the study launch command to recover without retraining.')}


def install(stage, payload):
    ledger = payload['ledger']
    config = ledger['plan']['base_config']
    workspace = config['remote']['workspace']
    root = root_for(workspace, ledger['study_id'])
    stage = Path(stage)
    if not stage.resolve().is_relative_to(Path(workspace) / 'study-staging'):
        raise ValueError('Controller stage is outside workspace')
    verify_snapshot(stage / 'project', payload['controller_manifest'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = (root / 'control.json').exists()
        if (root / 'study.json').exists():
            saved = study.read_json(root / 'study.json')
            if identity(saved) != identity(ledger):
                raise ValueError('Remote study differs from submitted immutable plan/source/IDs')
            # Remote state is authoritative once registered; stale clients never roll it back.
            if existing:
                ledger = saved
        if existing and alive(root / 'controller-launch.json'):
            return inspect(root)
        if not existing:
            source_manifest = payload['training_manifest']
            if source_manifest['source_sha256'] != ledger['source_sha256'] or source_manifest['files'] != ledger['source_files']:
                raise ValueError('Training source manifest differs from frozen study')
            origin_id = payload.get('source_run_id')
            if origin_id:
                if origin_id not in {c['run_id'] for c in ledger['children']}:
                    raise ValueError('Training source run is outside the study')
                parse_run_id(origin_id, remote.PACK)
                origin = Path(workspace) / 'runs' / origin_id
                actual = study.read_json(origin / 'manifest.json')
                if actual != source_manifest:
                    raise ValueError('Remote training source manifest differs from submitted evidence')
                source = origin / 'project'
            else:
                source = stage / 'project'
            target = root / 'training-source'
            if target.exists():
                verify_snapshot(target, source_manifest)
            else:
                copy_snapshot(source, target, source_manifest)
            remote.dump(root / 'training-manifest.json', source_manifest)
            validate(ledger, root)
            # Verify every existing result before authorizing any remaining child.
            for child in ledger['children']:
                report = child_status(workspace, child['run_id'])
                if report and child['status'] == 'PLANNED':
                    raise ValueError('Planned child already exists; ambiguous dispatch cannot be replaced')
                if child['status'] == 'PASS' and (not report or report['status'] != 'PASS'):
                    raise ValueError('Saved PASS child lacks completed remote evidence')
                if report and report['status'] == 'PASS':
                    verify_completed(ledger, child)
                    child['status'] = 'PASS'
            validate(ledger, root)
            remote.dump(root / 'study.json', ledger)
        sha = payload['controller_manifest']['source_sha256']
        controller = root / 'controllers' / sha / 'project'
        if controller.exists():
            verify_snapshot(controller, payload['controller_manifest'])
        else:
            copy_snapshot(stage / 'project', controller, payload['controller_manifest'])
        remote.dump(root / 'control.json', {'schema_version': 1, 'pack': remote.PACK,
                    'study_id': ledger['study_id'], 'study_identity': identity(ledger),
                    'controller_source_sha256': sha, 'controller_project': str(controller),
                    'controller_manifest': payload['controller_manifest'],
                    'training_source_sha256': ledger['source_sha256']})
        return start_locked(root)


def start_locked(root):
    if alive(root / 'controller-launch.json'):
        return inspect(root)
    ledger = study.read_json(root / 'study.json')
    if ledger['status'] == 'PASS':
        return inspect(root)
    if ledger['status'] == 'FAIL':
        raise ValueError('Failed study is not automatically rerun')
    # A lost launch reply must never replace a live controller with another PID.
    with (root / 'study.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Study is locked by a live coordinator with an unconfirmed launch identity') from exc
    verify_training_source(root, ledger)
    control = study.read_json(root / 'control.json')
    project = Path(control['controller_project'])
    verify_snapshot(project, control['controller_manifest'])
    config = ledger['plan']['base_config']['remote']
    lock_sha = remote.sha256(root / 'training-source/uv.lock')
    python = Path(config['workspace']) / 'environments' / lock_sha / '.venv/bin/python'
    if not python.is_file():
        raise ValueError('Frozen uv environment is absent; validate this source first')
    script = project / 'scripts/MoE_data_scale/localization_networks/workflow/study_worker.py'
    with (root / 'coordinator.log').open('ab') as log:
        process = subprocess.Popen([str(python), '-B', str(script), '--run', str(root)],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=project, start_new_session=True,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
    await_launch(process, root / 'controller-launch.json')
    return inspect(root)


def prepare_child(root, ledger, child):
    workspace = Path(ledger['plan']['base_config']['remote']['workspace'])
    run = workspace / 'runs' / child['run_id']
    # A durable attempt marker precedes all side effects. Ambiguous attempts are never repeated.
    if 'attempted_at' in child or run.exists():
        raise ValueError('Child was already attempted; refusing redispatch')
    source_manifest = verify_training_source(root, ledger)
    child.update(status='DISPATCHING', attempted_at=study.timestamp())
    study.save(root, ledger)
    run.mkdir(parents=True, exist_ok=False)
    config = child_config(ledger, child)
    suite, selection = suite_selection(ledger, child)
    copy_snapshot(root / 'training-source', run / 'project', source_manifest)
    (run / 'project/Origindata').symlink_to(Path(config['remote']['repository']) / 'Origindata', target_is_directory=True)
    for name in ('logs', 'reports', 'export/data', 'export/figure'):
        (run / name).mkdir(parents=True)
    request = {'schema_version': 1, 'pack': remote.PACK, 'run_id': run.name,
               'suite': suite, 'config': config, 'use_cache': True}
    manifest = {**source_manifest, 'run_id': run.name, 'suite': suite, 'created_at': study.timestamp(),
                'config_sha256': child['config_sha256']}
    if selection is not None:
        request['selection'] = selection
        manifest['selection'] = selection
    else:
        manifest.pop('selection', None)
    report = {'schema_version': 1, 'pack': remote.PACK, 'run_id': run.name, 'suite': suite,
              'status': 'STARTING', 'started_at': study.timestamp(), 'checks': [], 'remote_run_dir': str(run)}
    generation = {'schema_version': 1, 'dataset': config['dataset'], 'rbd': config['rbd'],
                  'result_path': 'outputs/validation/generation_result.json'}
    for name, value in [('request.json', request), ('manifest.json', manifest),
                        ('generation_config.json', generation), ('report.json', report)]:
        remote.dump(run / name, value)
    with (run / 'logs/worker.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--child', str(run)],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    await_launch(process, run / 'launch.json')
    print(f'Run: {run.name}', flush=True)


def launch_child(run):
    config = study.read_json(run / 'request.json')['config']
    project = run / 'project'
    workspace = Path(config['remote']['workspace'])
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'UV_PROJECT_ENVIRONMENT': str(workspace / 'environments' / remote.sha256(project / 'uv.lock') / '.venv'),
           'UV_CACHE_DIR': str(workspace / 'uv_cache'), 'UV_PYTHON_DOWNLOADS': 'never'}
    with (run / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (run / 'launch.json').exists():
            raise ValueError('Child launch already exists')
        command = [config['remote']['uv'], 'run', '--frozen', '--python', config['remote']['base_python'],
                   '--project', str(project), 'python', '-B',
                   str(project / 'scripts/MoE_data_scale/localization_networks/workflow/worker.py'), '--run-dir', str(run)]
        remote.dump(run / 'launch.json', process_identity(os.getpid()))
        os.chdir(project)
        os.execvpe(command[0], command, env)


def execute(root):
    with (root / 'study.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        remote.dump(root / 'controller-launch.json', process_identity(os.getpid()))
        ledger = study.read_json(root / 'study.json')
        validate(ledger, root)
        ledger['status'] = 'RUNNING'
        ledger.pop('error', None)
        study.save(root, ledger)
        workspace = ledger['plan']['base_config']['remote']['workspace']
        try:
            # Preflight every existing PASS before submitting anything new.
            for child in ledger['children']:
                if child['status'] == 'PASS':
                    verify_completed(ledger, child)
            for child in ledger['children']:
                if child['status'] == 'PASS':
                    continue
                report = child_status(workspace, child['run_id'])
                if child['status'] == 'PLANNED':
                    if report is not None:
                        raise ValueError('Unattempted child already exists; refusing duplicate launch')
                    prepare_child(root, ledger, child)
                elif report is None:
                    raise ValueError('Attempted child directory absent; refusing retraining')
                while True:
                    report = child_status(workspace, child['run_id'])
                    if report is None or report['status'] in {'UNKNOWN', 'FAIL'}:
                        child['status'] = report['status'] if report else 'UNKNOWN'
                        if child['status'] == 'FAIL':
                            ledger['status'] = 'FAIL'
                        raise RuntimeError(f"Child {child['run_id']} cannot continue: {report and report.get('error', report['status'])}")
                    if report['status'] == 'PASS':
                        verify_completed(ledger, child)
                        child['status'] = 'PASS'
                        child.pop('error', None)
                        study.save(root, ledger)
                        break
                    child['status'] = report['status']
                    study.save(root, ledger)
                    time.sleep(2)
            ledger['status'] = 'PASS'
            study.save(root, ledger)
            return 0
        except BaseException as exc:
            ledger.update(status='FAIL' if ledger['status'] == 'FAIL' else 'INTERRUPTED', error=f'{type(exc).__name__}: {exc}')
            study.save(root, ledger)
            print(ledger['error'], flush=True)
            return 1


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--child', type=Path)
    parser.add_argument('--rpc', choices=('install', 'status', 'snapshot'))
    args = parser.parse_args()
    if args.child is not None:
        return launch_child(args.child)
    if args.run is not None:
        return execute(args.run)
    payload = json.load(sys.stdin)
    root = root_for(payload['workspace'], payload['study_id'])
    if args.rpc == 'install':
        result = install(Path(payload['stage']), payload)
    else:
        result = inspect(root)
        if args.rpc == 'snapshot':
            result = {'report': result, 'ledger': study.read_json(root / 'study.json'),
                      'control': study.read_json(root / 'control.json')}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
