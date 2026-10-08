#!/usr/bin/env python3
"""Dispatch fixed source snapshots to haiqin; no local numerical imports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from config import RUN_ID, load_config

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("haiqin.toml")
LOCAL_RUNS = PROJECT / "outputs" / "remote_validation"
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
SOURCE_ROOTS = {"scripts", "simulation", "doc"}
SOURCE_FILES = {"README.md", "AGENTS.md", ".gitignore", ".gitattributes", "pyproject.toml", "uv.lock", ".python-version"}
EXCLUDED = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".DS_Store"}
TERMINAL = {"PASS", "FAIL"}


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths(project: Path = PROJECT) -> list[Path]:
    proc = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=project, check=True, capture_output=True)
    paths = []
    for raw in sorted(set(proc.stdout.split(b"\0")) - {b""}):
        name = raw.decode("utf-8")
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe source path: {name}")
        if rel.parts[0] not in SOURCE_ROOTS and name not in SOURCE_FILES:
            continue
        if any(part in EXCLUDED or part.startswith("._") for part in rel.parts) or rel.suffix in {".pyc", ".pyo", ".asv"} or rel.name.endswith("~"):
            continue
        path = project / rel
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != project and project in parent.parents):
            raise ValueError(f"Symlink source files are not supported: {name}")
        if path.exists() and path.is_file():
            paths.append(rel)
    if Path("scripts/localization_networks/workflow/worker.py") not in paths:
        raise ValueError("Snapshot does not contain the remote worker")
    return paths


def freeze_source(stage: Path, config: dict, run_id: str, suite: str, project: Path = PROJECT) -> dict:
    inventory = []
    for rel in source_paths(project):
        dest = stage / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / rel, dest)
        inventory.append({"path": rel.as_posix(), "size": dest.stat().st_size, "sha256": sha256(dest)})
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain=1"], cwd=project, check=True, capture_output=True, text=True).stdout.splitlines()
    return {"schema_version": 1, "run_id": run_id, "suite": suite, "created_at": datetime.now(timezone.utc).isoformat(), "git_head": head, "git_status": dirty, "source_sha256": digest, "files": inventory, "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}


def remote_python(config: dict, code: str, payload: dict | None = None) -> dict:
    remote = config["remote"]
    command = shlex.join([remote["base_python"], "-c", code])
    proc = subprocess.run(["ssh", *SSH_OPTIONS, remote["host"], command], input=json.dumps(payload or {}), text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"SSH command failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Expected remote JSON, received: {proc.stdout[:1000]}") from exc


CHECK_CODE = '''import json, pathlib, sys, subprocess
p=json.load(sys.stdin); r=p['remote']; root=pathlib.Path(r['repository']); raw=root/'Origindata'
uv=pathlib.Path(r['uv']); result={'repository':str(root),'repository_exists':root.is_dir(),'origindata_exists':raw.is_dir(),'base_python':sys.version.split()[0],'uv_exists':uv.is_file()}
if result['uv_exists']: result['uv_version']=subprocess.check_output([str(uv),'--version'],text=True).strip()
result['status']='PASS' if all([result['repository_exists'],result['origindata_exists'],result['uv_exists']]) else 'FAIL'
print(json.dumps(result))'''

INIT_CODE = '''import json, pathlib, sys, shutil
p=json.load(sys.stdin); workspace=pathlib.Path(p['workspace']); run=workspace/'runs'/p['run_id']; raw=pathlib.Path(p['repository'])/'Origindata'
if workspace.resolve()!=workspace: raise ValueError('Workspace may not contain symlink components')
if not raw.is_dir(): raise ValueError('Remote original data directory is absent')
run.mkdir(parents=True, exist_ok=False)
def write(name, value):
    path=run/name; temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_text(json.dumps(value)); temporary.replace(path)
write('dispatch.json', {'token':p['dispatch_token']})
write('report.json', p['report'])
try:
    for name in ['project','logs','reports','export']: (run/name).mkdir()
    for name in ['manifest.json','request.json','generation_config.json']: write(name,p[name])
    (run/'project'/'Origindata').symlink_to(raw, target_is_directory=True)
    for name in ['report.json','manifest.json','request.json','generation_config.json']: shutil.copyfile(run/name,run/'export'/name)
except BaseException as error:
    failure={**p['report'],'status':'FAIL','exit_code':1,'error':f'Remote initialization failed: {type(error).__name__}: {error}'}
    write('report.json',failure)
    raise
print(json.dumps({'run_dir':str(run)}))'''

DISPATCH_FAILURE_CODE = '''import json, pathlib, sys, shutil, fcntl
p=json.load(sys.stdin); run=pathlib.Path(p['run_dir']); marker=run/'dispatch.json'
if not marker.is_file() or json.loads(marker.read_text()).get('token')!=p['dispatch_token']:
    print(json.dumps({'marked':False,'reason':'Run was not created by this dispatch'})); sys.exit(0)
lock=(run/'dispatch.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX)
report=run/'report.json'; current=json.loads(report.read_text()) if report.is_file() else {'status':'STARTING'}
if (run/'launch.json').exists() or current.get('status')!='STARTING':
    print(json.dumps({'marked':False,'report':current})); sys.exit(0)
failure=p['report']; temporary=report.with_suffix('.json.tmp'); temporary.write_text(json.dumps(failure)); temporary.replace(report)
export=run/'export'; export.mkdir(exist_ok=True)
for name in ['report.json','manifest.json','request.json','generation_config.json']:
    if (run/name).is_file(): shutil.copyfile(run/name,export/name)
print(json.dumps({'marked':True,'report':failure}))'''

START_CODE = '''import json, pathlib, sys, subprocess, os, fcntl, hashlib
p=json.load(sys.stdin); run=pathlib.Path(p['run_dir']); project=run/'project'
lock=(run/'dispatch.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX)
for n in ['manifest.json','request.json','generation_config.json']: (project/n).replace(run/n)
lock_hash=hashlib.sha256((project/'uv.lock').read_bytes()).hexdigest()
venv=pathlib.Path(p['workspace'])/'environments'/lock_hash/'.venv'
worker_env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','UV_PROJECT_ENVIRONMENT':str(venv),'UV_CACHE_DIR':str(pathlib.Path(p['workspace'])/'uv_cache'),'UV_PYTHON_DOWNLOADS':'never'}
with (run/'logs'/'worker.log').open('ab') as log:
    child=subprocess.Popen([p['uv'],'run','--frozen','--python',p['base_python'],'--project',str(project),'python','-B',str(project/'scripts/localization_networks/workflow/worker.py'),'--run-dir',str(run)],cwd=project,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=worker_env)
identity={'pid':child.pid,'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':pathlib.Path(f'/proc/{child.pid}/stat').read_text().split()[21]}
launch=run/'launch.json'; temporary=run/'launch.json.tmp'; temporary.write_text(json.dumps(identity)); temporary.replace(launch)
print(json.dumps({'run_dir':str(run),**identity}))'''

STATUS_CODE = '''import json, pathlib, sys
p=json.load(sys.stdin); run=pathlib.Path(p['run_dir']); report=run/'report.json'
if not run.is_dir(): raise FileNotFoundError(str(run))
result=json.loads(report.read_text()) if report.exists() else {'run_id':run.name,'status':'STARTING','checks':[]}
launch=run/'launch.json'
if result['status'] not in ['PASS','FAIL'] and launch.exists():
    ident=json.loads(launch.read_text()); proc=pathlib.Path(f"/proc/{ident['pid']}/stat")
    alive=False
    try:
        parts=proc.read_text().split(); alive=parts[2]!='Z' and parts[21]==ident['start_ticks'] and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==ident['boot_id']
    except FileNotFoundError: pass
    if not alive: result.update(status='FAIL',exit_code=1,error='Remote worker stopped without a terminal report; inspect logs/worker.log')
print(json.dumps(result))'''


def run_dir(config: dict, run_id: str) -> str:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid run ID; use the ID printed by validate")
    return f"{config['remote']['workspace']}/runs/{run_id}"


def status(config: dict, run_id: str) -> dict:
    return remote_python(config, STATUS_CODE, {"run_dir": run_dir(config, run_id)})


def summarize(report: dict) -> None:
    print(f"{report['run_id']}: {report['status']}", flush=True)
    for row in report.get("checks", []):
        print(f"  {row['status']:8s} {row['name']}" + (f": {row['detail']}" if row.get("detail") else ""), flush=True)
    if report.get("error"):
        print(report["error"], file=sys.stderr)


def fetch(config: dict, run_id: str) -> dict:
    report = status(config, run_id)
    local = LOCAL_RUNS / run_id
    local.mkdir(parents=True, exist_ok=True)
    dump(local / "report.json", report)
    # Copy only the worker's explicit small-artifact export; HDF5/PT are excluded there.
    source = f"{config['remote']['host']}:{run_dir(config, run_id)}/export/"
    proc = subprocess.run(["rsync", "-rlt", "-e", shlex.join(["ssh", *SSH_OPTIONS]), source, str(local) + "/"], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"Result retrieval failed: {proc.stderr.strip()}")
    # Live/crashed workers may not have exported anything yet; always retain logs.
    proc = subprocess.run(["rsync", "-rlt", "-e", shlex.join(["ssh", *SSH_OPTIONS]), f"{config['remote']['host']}:{run_dir(config, run_id)}/logs/", str(local / "logs") + "/"], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"Log retrieval failed: {proc.stderr.strip()}")
    dump(local / "report.json", report)
    print(f"Results: {local}")
    return report


def validate(config: dict, suite: str, detach: bool, dry_run: bool, no_cache: bool) -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    remote = config["remote"]
    remote_dir = run_dir(config, run_id)
    request = {"schema_version": 1, "run_id": run_id, "suite": suite, "config": config, "use_cache": not no_cache}
    generation_config = {"schema_version": 1, "dataset": config["dataset"], "rbd": config["rbd"], "result_path": "outputs/validation/generation_result.json"}
    local = LOCAL_RUNS / run_id
    dispatch_token = uuid.uuid4().hex
    starting = {"schema_version": 1, "run_id": run_id, "suite": suite, "status": "STARTING", "started_at": datetime.now(timezone.utc).isoformat(), "checks": [], "remote_run_dir": remote_dir}
    if not dry_run:
        # A collision never overwrites an earlier run or its evidence.
        local.mkdir(parents=True, exist_ok=False)
        dump(local / "request.json", request)
        dump(local / "generation_config.json", generation_config)
        dump(local / "dispatch.json", {"token": dispatch_token})
        dump(local / "report.json", starting)
        print(f"Run: {run_id}", flush=True)
    with tempfile.TemporaryDirectory(prefix="swellex-source-") as temp:
        stage = Path(temp)
        init_attempted = False
        try:
            manifest = freeze_source(stage, config, run_id, suite)
            if dry_run:
                print(json.dumps({"run_id": run_id, "remote_run_dir": remote_dir, "files": len(manifest["files"]), "source_sha256": manifest["source_sha256"], "request": request, "generation_config": generation_config}, ensure_ascii=False, indent=2))
                return 0
            dump(local / "manifest.json", manifest)
            dump(stage / "manifest.json", manifest)
            dump(stage / "request.json", request)
            dump(stage / "generation_config.json", generation_config)
            preflight = remote_python(config, CHECK_CODE, config)
            dump(local / "preflight.json", preflight)
            if preflight["status"] != "PASS":
                raise RuntimeError(f"Remote environment check failed: {preflight}")
            init_attempted = True
            initialized = remote_python(config, INIT_CODE, {"workspace": remote["workspace"], "repository": remote["repository"], "run_id": run_id, "dispatch_token": dispatch_token, "report": starting, "manifest.json": manifest, "request.json": request, "generation_config.json": generation_config})
            if initialized["run_dir"] != remote_dir:
                raise RuntimeError("Remote initialization returned an unexpected path")
            proc = subprocess.run(["rsync", "-rlt", "-e", shlex.join(["ssh", *SSH_OPTIONS]), str(stage) + "/", f"{remote['host']}:{remote_dir}/project/"], capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError(f"Source transfer failed: {proc.stderr.strip()}")
            launched = remote_python(config, START_CODE, {"run_dir": remote_dir, "base_python": remote["base_python"], "uv": remote["uv"], "workspace": remote["workspace"]})
            dump(local / "launch.json", launched)
        except BaseException as exc:
            if not dry_run:
                failure = {**starting, "status": "FAIL", "exit_code": 130 if isinstance(exc, KeyboardInterrupt) else 1, "finished_at": datetime.now(timezone.utc).isoformat(), "error": f"Dispatch failed: {type(exc).__name__}: {exc}"}
                dump(local / "report.json", failure)
                if init_attempted:
                    try:
                        result = remote_python(config, DISPATCH_FAILURE_CODE, {"run_dir": remote_dir, "dispatch_token": dispatch_token, "report": failure})
                        # An ambiguous SSH launch failure may leave a healthy
                        # worker running; never overwrite its authoritative report.
                        failure["remote_recovery"] = result
                    except Exception as recovery:
                        failure["remote_recovery_error"] = f"{type(recovery).__name__}: {recovery}"
                    dump(local / "report.json", failure)
                print(f"Dispatch evidence: {local}", file=sys.stderr)
            raise
        if detach:
            print(f"Detached. Inspect with: uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py status --run {run_id}")
            return 0
        previous = None
        try:
            while True:
                report = status(config, run_id)
                progress = (report["status"], len(report.get("checks", [])))
                if progress != previous:
                    summarize(report)
                    previous = progress
                if report["status"] in TERMINAL:
                    fetch(config, run_id)
                    return 0 if report["status"] == "PASS" else int(report.get("exit_code", 1)) or 1
                time.sleep(5)
        except KeyboardInterrupt:
            print(f"Remote task continues. Run ID: {run_id}")
            return 130


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Check haiqin dependencies and GPU without launching MATLAB")
    validate_parser = sub.add_parser("validate", help="Freeze source and run validation on haiqin")
    validate_parser.add_argument("--suite", choices=["quick", "smoke", "experiment"], default="smoke")
    validate_parser.add_argument("--detach", action="store_true")
    validate_parser.add_argument("--dry-run", action="store_true")
    validate_parser.add_argument("--no-cache", action="store_true")
    for name in ("status", "fetch"):
        sub.add_parser(name).add_argument("--run", required=True)
    args = parser.parse_args()
    try:
        config = load_config(args.config.resolve())
        if args.command == "check":
            result = remote_python(config, CHECK_CODE, config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "PASS" else 1
        if args.command == "validate":
            return validate(config, args.suite, args.detach, args.dry_run, args.no_cache)
        report = status(config, args.run) if args.command == "status" else fetch(config, args.run)
        summarize(report)
        return 0 if report["status"] == "PASS" else (1 if report["status"] == "FAIL" else 2)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
