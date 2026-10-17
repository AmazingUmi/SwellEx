#!/usr/bin/env python3
"""Dispatch isolated continual MoE demonstration on haiqin; stdlib only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid

PROJECT = Path(__file__).resolve().parents[4]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts.MoE_learn.localization_networks.workflow.config import load_config, validate_config
from scripts.run_identity import make_run_id, parse_run_id, validate_filters, validate_run_identity
from scripts.run_monitor import LOG_PROGRESS_CODE, RemoteConnectionError, follow
from scripts.MoE_learn.localization_networks.workflow.tasks import ARCHITECTURES, METHODS, model_plan, resolved_config

DEFAULT_CONFIG = Path(__file__).with_name("haiqin_demo.toml")
PACK = "MoE_learn"
LOCAL_RUNS = PROJECT / "output" / "data"
LOCAL_FIGURES = PROJECT / "output" / "figure"
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
SOURCE_ROOTS = {"scripts", "simulation", "doc"}
SOURCE_FILES = {"README.md", "AGENTS.md", ".gitignore", ".gitattributes", "pyproject.toml", "uv.lock", ".python-version"}
EXCLUDED = {"output", "outputs", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".DS_Store"}
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
    if Path("scripts/MoE_learn/localization_networks/workflow/worker.py") not in paths:
        raise ValueError("Snapshot does not contain the remote worker")
    return paths


def freeze_source(stage: Path, config: dict, run_id: str, suite: str, project: Path = PROJECT, *, selection: dict | None = None) -> dict:
    model_plan(suite, selection)  # Validate the complete selection before any side effects.
    inventory = []
    for rel in source_paths(project):
        dest = stage / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / rel, dest)
        shutil.copymode(project / rel, dest)
        inventory.append({"path": rel.as_posix(), "size": dest.stat().st_size, "sha256": sha256(dest)})
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain=1"], cwd=project, check=True, capture_output=True, text=True).stdout.splitlines()
    manifest = {"schema_version": 1, "pack": PACK, "run_id": run_id, "suite": suite, "created_at": datetime.now(timezone.utc).isoformat(), "git_head": head, "git_status": dirty, "source_sha256": digest, "files": inventory, "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    if suite == "selected":
        manifest["selection"] = dict(selection)
    return manifest


def remote_python(config: dict, code: str, payload: dict | None = None) -> dict:
    remote = config["remote"]
    command = shlex.join([remote["base_python"], "-c", code])
    proc = subprocess.run(["ssh", *SSH_OPTIONS, remote["host"], command], input=json.dumps(payload or {}), text=True, capture_output=True)
    if proc.returncode:
        error = RemoteConnectionError if proc.returncode == 255 else RuntimeError
        raise error(f"SSH command failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Expected remote JSON, received: {proc.stdout[:1000]}") from exc


CHECK_CODE = '''import json, pathlib, sys, subprocess
p=json.load(sys.stdin); r=p['remote']; root=pathlib.Path(r['repository']); raw=root/'Origindata'
uv=pathlib.Path(r['uv']); result={'repository':str(root),'repository_exists':root.is_dir(),'origindata_exists':raw.is_dir(),'base_python':sys.version.split()[0],'uv_exists':uv.is_file()}
if result['uv_exists']: result['uv_version']=subprocess.check_output([str(uv),'--version'],text=True).strip()
result['status']='PASS' if all([result['repository_exists'],result['uv_exists'],not p.get('require_raw',True) or result['origindata_exists']]) else 'FAIL'
print(json.dumps(result))'''

INIT_CODE = '''import json, pathlib, sys, shutil
p=json.load(sys.stdin); workspace=pathlib.Path(p['workspace']); run=workspace/'runs'/p['run_id']; raw=pathlib.Path(p['repository'])/'Origindata'
if workspace.resolve()!=workspace: raise ValueError('Workspace may not contain symlink components')
if p['request.json']['suite']!='quick' and not raw.is_dir(): raise ValueError('Remote original data directory is absent')
run.mkdir(parents=True, exist_ok=False)
def write(name, value):
    path=run/name; temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_text(json.dumps(value)); temporary.replace(path)
write('dispatch.json', {'token':p['dispatch_token']})
write('report.json', p['report'])
try:
    for name in ['project','logs','reports','export']: (run/name).mkdir()
    for name in ['data','figure']: (run/'export'/name).mkdir()
    for name in ['manifest.json','request.json','resolved-config.json']: write(name,p[name])
    if raw.is_dir(): (run/'project'/'Origindata').symlink_to(raw, target_is_directory=True)
    for name in ['report.json','manifest.json','request.json','resolved-config.json']: shutil.copyfile(run/name,run/'export'/'data'/name)
except BaseException as error:
    failure={**p['report'],'status':'FAIL','exit_code':1,'error':f'Remote initialization failed: {type(error).__name__}: {error}'}
    write('report.json',failure)
    (run/'export'/'data').mkdir(parents=True,exist_ok=True)
    (run/'export'/'figure').mkdir(parents=True,exist_ok=True)
    for name in ['report.json','manifest.json','request.json','resolved-config.json']:
        if (run/name).is_file(): shutil.copyfile(run/name,run/'export'/'data'/name)
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
export=run/'export'/'data'; export.mkdir(parents=True,exist_ok=True)
(run/'export'/'figure').mkdir(parents=True,exist_ok=True)
for name in ['report.json','manifest.json','request.json','resolved-config.json']:
    if (run/name).is_file(): shutil.copyfile(run/name,export/name)
print(json.dumps({'marked':True,'report':failure}))'''

START_CODE = '''import json, pathlib, sys, subprocess, os, fcntl, hashlib
p=json.load(sys.stdin); run=pathlib.Path(p['run_dir']); project=run/'project'
lock=(run/'dispatch.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX)
for n in ['manifest.json','request.json','resolved-config.json']: (project/n).replace(run/n)
lock_hash=hashlib.sha256((project/'uv.lock').read_bytes()).hexdigest()
venv=pathlib.Path(p['workspace'])/'environments'/lock_hash/'.venv'
worker_env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','UV_PROJECT_ENVIRONMENT':str(venv),'UV_CACHE_DIR':str(pathlib.Path(p['workspace'])/'uv_cache'),'UV_PYTHON_DOWNLOADS':'never'}
with (run/'logs'/'worker.log').open('ab') as log:
    child=subprocess.Popen([p['uv'],'run','--frozen','--python',p['base_python'],'--project',str(project),'python','-B',str(project/'scripts/MoE_learn/localization_networks/workflow/worker.py'),'--run-dir',str(run)],cwd=project,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=worker_env)
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
    parse_run_id(run_id, PACK)
    return f"{config['remote']['workspace']}/runs/{run_id}"


def status(config: dict, run_id: str) -> dict:
    report = remote_python(config, STATUS_CODE, {"run_dir": run_dir(config, run_id)})
    if report.get("pack") != PACK or report.get("run_id") != run_id:
        raise ValueError(f"Run identity mismatch: expected {PACK}/{run_id}")
    return report


def progress(config: dict, run_id: str, offsets: dict) -> dict:
    code = STATUS_CODE.replace("print(json.dumps(result))", LOG_PROGRESS_CODE)
    update = remote_python(config, code, {"run_dir": run_dir(config, run_id), "offsets": offsets})
    report = update["report"]
    if report.get("pack") != PACK or report.get("run_id") != run_id:
        raise ValueError(f"Run identity mismatch: expected {PACK}/{run_id}")
    return update


def follow_run(config: dict, run_id: str, *, max_connection_failures: int | None = None) -> int:
    if max_connection_failures is not None and (type(max_connection_failures) is not int or max_connection_failures < 1):
        raise ValueError("max_connection_failures must be a positive integer")
    failures = {"progress": 0, "fetch": 0}

    def bounded_call(kind, operation):
        try:
            result = operation()
        except (RemoteConnectionError, OSError) as exc:
            failures[kind] += 1
            if max_connection_failures is not None and failures[kind] >= max_connection_failures:
                raise RuntimeError(f"{kind} failed {failures[kind]} consecutive times; check status and fetch results for {run_id}") from exc
            raise
        failures[kind] = 0
        return result

    return follow(run_id, progress=lambda offsets: bounded_call("progress", lambda: progress(config, run_id, offsets)),
                  fetch=lambda: bounded_call("fetch", lambda: fetch(config, run_id)), summarize=summarize)


def summarize(report: dict) -> None:
    print(f"{report['run_id']}: {report['status']}", flush=True)
    for row in report.get("checks", []):
        print(f"  {row['status']:8s} {row['name']}" + (f": {row['detail']}" if row.get("detail") else ""), flush=True)
    if report.get("error"):
        print(report["error"], file=sys.stderr)


def verify_artifact_delivery(local: Path, figures: Path, run_id: str) -> dict | None:
    """Verify the exact exported demo artifacts, including its checkpoint."""
    path = local / "artifact-manifest.json"
    if path.is_symlink():
        raise ValueError("External artifact delivery manifest")
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("pack") != PACK or manifest.get("run_id") != run_id
            or not isinstance(manifest.get("files"), list) or not manifest["files"]):
        raise ValueError("Invalid artifact delivery manifest")
    roots = {"data": local, "figure": figures}
    seen = set()
    for item in manifest["files"]:
        if not isinstance(item, dict) or item.get("group") not in roots or not isinstance(item.get("path"), str):
            raise ValueError("Invalid artifact delivery entry")
        rel = PurePosixPath(item["path"])
        identity = (item["group"], str(rel))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or str(rel) != item["path"] or identity in seen:
            raise ValueError("Unsafe or duplicate artifact delivery path")
        seen.add(identity)
        root = roots[item["group"]]
        artifact = root / str(rel)
        if (type(item.get("size")) is not int or item["size"] < 0
                or not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])
                or not artifact.is_file() or artifact.is_symlink() or not artifact.resolve().is_relative_to(root.resolve())
                or artifact.stat().st_size != item["size"] or sha256(artifact) != item["sha256"]):
            raise ValueError(f"Artifact delivery hash mismatch: {identity}")
    return {"schema_version": 1, "status": "PASS", "run_id": run_id,
            "scope": "demo_artifacts_including_checkpoint", "files_verified": len(seen),
            "manifest_sha256": sha256(path)}


def local_paths(run_id: str, *, config: dict | None = None) -> tuple[Path, Path]:
    parse_run_id(run_id, PACK)
    paths = (LOCAL_RUNS / run_id, LOCAL_FIGURES / run_id)
    for path in paths:
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError(f"Symlink local delivery path: {path}")
    return paths


def fetch(config: dict, run_id: str) -> dict:
    initial_report = status(config, run_id)
    local, figures = local_paths(run_id, config=config)
    local.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    dump(local / "report.json", initial_report)
    # Copy the explicit export; generated HDF5 stays remote, the demo checkpoint is included.
    def retrieve_export():
        for kind, destination in (("data", local), ("figure", figures)):
            source = f"{config['remote']['host']}:{run_dir(config, run_id)}/export/{kind}/"
            proc = subprocess.run(["rsync", "-rlt", "--checksum", "-e", shlex.join(["ssh", *SSH_OPTIONS]), source, str(destination) + "/"], capture_output=True, text=True)
            if proc.returncode:
                error = RemoteConnectionError if proc.returncode in {10, 12, 30, 35, 255} else RuntimeError
                raise error(f"Result {kind} retrieval failed: {proc.stderr.strip()}")
    retrieve_export()
    # Live/crashed workers may not have exported anything yet; always retain logs.
    proc = subprocess.run(["rsync", "-rlt", "--checksum", "-e", shlex.join(["ssh", *SSH_OPTIONS]), f"{config['remote']['host']}:{run_dir(config, run_id)}/logs/", str(local / "logs") + "/"], capture_output=True, text=True)
    if proc.returncode:
        error = RemoteConnectionError if proc.returncode in {10, 12, 30, 35, 255} else RuntimeError
        raise error(f"Log retrieval failed: {proc.stderr.strip()}")
    report = status(config, run_id)
    # Once terminal, export is immutable. If completion occurred during the
    # transfer, retrieve it again before reporting or verifying that completion.
    if initial_report["status"] not in TERMINAL and report["status"] in TERMINAL:
        retrieve_export()
    dump(local / "report.json", report)
    if report["status"] in TERMINAL:
        try:
            verification = verify_artifact_delivery(local, figures, run_id)
            if verification is None and report["status"] == "PASS":
                raise ValueError("Completed demo run is missing its artifact delivery manifest")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            dump(local / "delivery-verification.json", {"schema_version": 1, "status": "FAIL", "run_id": run_id,
                                                       "scope": "demo_artifacts_including_checkpoint", "error": str(exc)})
            raise
        if verification is not None:
            dump(local / "delivery-verification.json", verification)
    print(f"Results: {local}")
    print(f"Figures: {figures}")
    return report


def dispatch(config: dict, suite: str, no_cache: bool, *, selection: dict | None = None,
             run_id: str | None = None, expected_source_sha256: str | None = None,
             max_connection_failures: int | None = None) -> int:
    config = validate_config(config)
    model_plan(suite, selection)  # Validate the complete selection before any side effects.
    run_id = make_run_id(PACK, config, selection) if run_id is None else run_id
    validate_run_identity(run_id, PACK, config, selection)
    if expected_source_sha256 is not None and not re.fullmatch(r"[a-f0-9]{64}", expected_source_sha256):
        raise ValueError("expected_source_sha256 must be a SHA256 digest")
    if max_connection_failures is not None and (type(max_connection_failures) is not int or max_connection_failures < 1):
        raise ValueError("max_connection_failures must be a positive integer")
    remote = config["remote"]
    remote_dir = run_dir(config, run_id)
    request = {"schema_version": 1, "pack": PACK, "run_id": run_id, "suite": suite, "config": config, "use_cache": not no_cache}
    if suite == "selected":
        request["selection"] = dict(selection)
    execution_config = resolved_config(config, suite)
    local, _ = local_paths(run_id, config=config)
    dispatch_token = uuid.uuid4().hex
    starting = {"schema_version": 1, "pack": PACK, "run_id": run_id, "suite": suite, "status": "STARTING", "started_at": datetime.now(timezone.utc).isoformat(), "checks": [], "remote_run_dir": remote_dir}
    if suite == "selected":
        starting["selection"] = dict(selection)
    # A collision never overwrites an earlier run or its evidence.
    local.mkdir(parents=True, exist_ok=False)
    dump(local / "request.json", request)
    dump(local / "resolved-config.json", execution_config)
    dump(local / "dispatch.json", {"token": dispatch_token})
    dump(local / "report.json", starting)
    print(f"Run: {run_id}", flush=True)
    with tempfile.TemporaryDirectory(prefix="swellex-source-") as temp:
        stage = Path(temp)
        init_attempted = False
        try:
            manifest = freeze_source(stage, config, run_id, suite, selection=selection) if suite == "selected" else freeze_source(stage, config, run_id, suite)
            if expected_source_sha256 is not None and manifest["source_sha256"] != expected_source_sha256:
                raise ValueError("Source differs from the requested frozen identity; run was not launched")
            dump(local / "manifest.json", manifest)
            dump(stage / "manifest.json", manifest)
            dump(stage / "request.json", request)
            dump(stage / "resolved-config.json", execution_config)
            preflight = remote_python(config, CHECK_CODE, {**config, "require_raw": suite != "quick"})
            dump(local / "preflight.json", preflight)
            if preflight["status"] != "PASS":
                raise RuntimeError(f"Remote environment check failed: {preflight}")
            init_attempted = True
            initialized = remote_python(config, INIT_CODE, {"workspace": remote["workspace"], "repository": remote["repository"], "run_id": run_id, "dispatch_token": dispatch_token, "report": starting, "manifest.json": manifest, "request.json": request, "resolved-config.json": execution_config})
            if initialized["run_dir"] != remote_dir:
                raise RuntimeError("Remote initialization returned an unexpected path")
            proc = subprocess.run(["rsync", "-rltp", "-e", shlex.join(["ssh", *SSH_OPTIONS]), str(stage) + "/", f"{remote['host']}:{remote_dir}/project/"], capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError(f"Source transfer failed: {proc.stderr.strip()}")
            launched = remote_python(config, START_CODE, {"run_dir": remote_dir, "base_python": remote["base_python"], "uv": remote["uv"], "workspace": remote["workspace"]})
            dump(local / "launch.json", launched)
        except BaseException as exc:
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
    if max_connection_failures is None:
        return follow_run(config, run_id)
    return follow_run(config, run_id, max_connection_failures=max_connection_failures)


def validate(config: dict, suite: str, no_cache: bool) -> int:
    if suite not in {"quick", "smoke", "experiment"}:
        raise ValueError("Validation suite must be quick, smoke, or experiment")
    return dispatch(config, suite, no_cache)


def selected_run(config: dict, method: str | None, network: str | None, no_cache: bool) -> int:
    selection = validate_filters(PACK, {"method": method, "network": network})
    return dispatch(config, "selected", no_cache, selection=selection)


def make_parser() -> argparse.ArgumentParser:
    from scripts.pack_cli import make_pack_parser
    return make_pack_parser(PACK, DEFAULT_CONFIG, METHODS, ARCHITECTURES)


def dry_run(config: dict, suite: str, selection: dict | None = None) -> int:
    plan = model_plan(suite, selection)
    effective = resolved_config(config, suite)
    print(json.dumps({"pack": PACK, "status": "NOT_RUN", "suite": suite,
                      "purpose": "synthetic_cuda_validation" if suite == "quick" else "continual_learning_demo",
                      "model_plan": plan, "config": effective,
                      "cache_policy": "fresh_run_no_dataset_reuse",
                      "worker_module": "scripts.MoE_learn.localization_networks.demo"},
                     ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        from scripts.pack_cli import config_path
        config = load_config(config_path(args).resolve())
        if args.command == "check":
            result = remote_python(config, CHECK_CODE, config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "PASS" else 1
        if args.command == "validate":
            return dry_run(config, args.suite) if args.dry_run else validate(config, args.suite, args.no_cache)
        if args.command is None:
            selection = None
            if args.method is not None or args.network is not None:
                selection = validate_filters(PACK, {"method": args.method, "network": args.network})
            suite = "selected" if selection else "experiment"
            if args.dry_run:
                return dry_run(config, suite, selection)
            if selection:
                return dispatch(config, suite, args.no_cache, selection=selection)
            return dispatch(config, suite, args.no_cache)
        report = status(config, args.run) if args.command == "status" else fetch(config, args.run)
        summarize(report)
        return 0 if report["status"] == "PASS" else (1 if report["status"] == "FAIL" else 2)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
