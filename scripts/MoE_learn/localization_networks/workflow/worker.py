"""Frozen remote execution of the small continual MoE demonstration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[4]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts.MoE_learn.localization_networks.workflow.config import validate_config
from scripts.MoE_learn.localization_networks.workflow.processes import run_process
from scripts.MoE_learn.localization_networks.workflow.tasks import SUITES, expected_phases, model_plan, resolved_config
from scripts.run_identity import validate_filters, validate_run_identity

PACK = "MoE_learn"


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(project: Path, manifest: dict) -> None:
    inventory = manifest["files"]
    seen = set()
    for item in inventory:
        rel = PurePosixPath(item["path"])
        if rel.is_absolute() or ".." in rel.parts or str(rel) in seen:
            raise ValueError(f"Invalid/duplicate manifest path: {rel}")
        seen.add(str(rel))
        path = project / str(rel)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(project.resolve()):
            raise ValueError(f"Snapshot contains missing or external source: {rel}")
        if path.stat().st_size != item["size"] or file_hash(path) != item["sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {rel}")
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != manifest["source_sha256"]:
        raise ValueError("Source manifest digest mismatch")
    actual = {p.relative_to(project).as_posix() for p in project.rglob("*") if p.is_file() and not {"output", "outputs", "Origindata", "__pycache__"}.intersection(p.relative_to(project).parts)}
    if actual != seen:
        raise ValueError(f"Unexpected source inventory: {sorted(actual ^ seen)}")


def workspace_environment(config: dict, lock_hash: str) -> Path:
    return Path(config["remote"]["workspace"]) / "environments" / lock_hash / ".venv"


class Worker:
    def __init__(self, run: Path):
        self.run = run.resolve()
        self.project = self.run / "project"
        self.report = {"schema_version": 1, "pack": PACK, "run_id": self.run.name, "status": "RUNNING", "started_at": stamp(), "checks": [], "purpose": "engineering_validation"}
        self.config = None
        self.env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}

    def save(self) -> None:
        self.report["updated_at"] = stamp()
        write_json(self.run / "report.json", self.report)

    def phase(self, name: str, action) -> object:
        row = {"name": name, "status": "RUNNING", "started_at": stamp()}
        self.report["checks"].append(row)
        self.save()
        started = time.monotonic()
        print(f"START {name}", flush=True)
        try:
            value = action()
            row.update(status="PASS", elapsed_s=round(time.monotonic()-started, 3))
            if isinstance(value, dict):
                row["result"] = value
            print(f"PASS {name}", flush=True)
            return value
        except BaseException as exc:
            row.update(status="FAIL", detail=str(exc), elapsed_s=round(time.monotonic()-started, 3))
            raise
        finally:
            row["finished_at"] = stamp()
            self.save()

    def command(self, name: str, argv: list[str], timeout: int, *, env: dict | None = None) -> None:
        log_path = self.run / "logs" / f"{name}.log"
        write_json(self.run / "logs" / f"{name}.command.json", {"argv": argv, "cwd": str(self.project), "timeout_s": timeout})
        with log_path.open("wb") as log:
            result = run_process(argv, cwd=self.project, env=self.env if env is None else env,
                                 stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
        if code:
            tail = log_path.read_text(errors="replace")[-2500:]
            raise RuntimeError(f"{name} exited {code}; see {log_path}\n{tail}")

    def json_command(self, name: str, argv: list[str], report_path: Path, timeout: int, *, env: dict | None = None) -> dict:
        self.command(name, argv, timeout, env=env)
        result = json.loads(report_path.read_text())
        if result.get("status") != "PASS":
            raise RuntimeError(f"{name} did not produce a PASS report: {result}")
        return result

    def setup(self) -> dict:
        request = json.loads((self.run / "request.json").read_text())
        if not isinstance(request, dict):
            raise ValueError("Invalid remote request schema")
        fields = {"schema_version", "pack", "run_id", "suite", "config", "use_cache"}
        if request.get("suite") == "selected":
            fields.add("selection")
        if set(request) != fields or type(request["schema_version"]) is not int or request["schema_version"] != 1:
            raise ValueError("Invalid remote request schema")
        if request["pack"] != PACK:
            raise ValueError("Remote request pack mismatch")
        if request["run_id"] != self.run.name or request["suite"] not in SUITES or not isinstance(request["use_cache"], bool):
            raise ValueError("Invalid run ID, suite or cache policy")
        if request["suite"] == "selected":
            validate_filters(PACK, request["selection"])
        self.request = request
        self.config = validate_config(request["config"])
        validate_run_identity(self.run.name, PACK, self.config, request.get("selection"))
        expected = Path(self.config["remote"]["workspace"]) / "runs" / self.run.name
        if self.run != expected or self.project.resolve() != self.project:
            raise ValueError("Remote run is outside configured snapshot location")
        manifest = json.loads((self.run / "manifest.json").read_text())
        if manifest.get("pack") != PACK or manifest["run_id"] != self.run.name or manifest["suite"] != request["suite"]:
            raise ValueError("Manifest pack/run/suite mismatch")
        if request["suite"] == "selected":
            selection = validate_filters(PACK, manifest.get("selection"))
            if selection != request["selection"]:
                raise ValueError("Manifest selection differs from the frozen request")
        elif "selection" in manifest:
            raise ValueError("Validation manifests cannot contain a selection")
        config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        if config_hash != manifest["config_sha256"]:
            raise ValueError("Request configuration hash mismatch")
        verify_snapshot(self.project, manifest)
        execution_config = json.loads((self.run / "resolved-config.json").read_text())
        expected_config = resolved_config(self.config, request["suite"])
        if execution_config != expected_config:
            raise ValueError("Resolved execution configuration differs from the frozen request")
        self.execution_config = execution_config
        self.manifest = manifest
        self.report.update(suite=request["suite"], source_sha256=manifest["source_sha256"], git_head=manifest["git_head"], config_sha256=config_hash,
                           expert_policy_status="demo_uncalibrated",
                           resolved_config_sha256=file_hash(self.run / "resolved-config.json"),
                           cache_policy="fresh_run_no_dataset_reuse",
                           model_plan=model_plan(request["suite"], request.get("selection")))
        if request["suite"] == "experiment":
            self.report["purpose"] = "continual_learning_demo_not_scientific_performance_evidence"
        elif request["suite"] == "selected":
            self.report.update(selection=dict(request["selection"]), purpose="continual_learning_demo_not_scientific_performance_evidence")
        if request["suite"] == "quick":
            self.report["purpose"] = "synthetic_cuda_engineering_validation"
        threads = str(self.config["validation"]["threads"])
        self.env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
        self.python = sys.executable
        self.lock_hash = file_hash(self.project / "uv.lock")
        expected_environment = workspace_environment(self.config, self.lock_hash)
        if Path(sys.prefix).resolve() != expected_environment.resolve():
            raise ValueError("Worker must run inside the uv environment for the frozen lockfile")
        self.report["uv_lock_sha256"] = self.lock_hash
        return {"files": len(manifest["files"]), "source_sha256": manifest["source_sha256"]}

    def environment(self) -> dict:
        import torch
        info = {"python": sys.version, "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "scipy", "h5py", "matplotlib")}, "cuda_available": torch.cuda.is_available()}
        device = torch.device(self.config["validation"]["device"])
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA validation requires an available GPU")
            info["gpu"] = torch.cuda.get_device_name(device)
        info.update(uv_lock_sha256=self.lock_hash, uv_environment=str(Path(sys.prefix)), uv_version=subprocess.check_output([self.config["remote"]["uv"], "--version"], text=True).strip(), operating_system=platform.freedesktop_os_release().get("PRETTY_NAME"), libc=platform.libc_ver())
        write_json(self.run / "environment.json", info)
        return info

    def data_identity(self) -> dict:
        raw = Path(self.config["remote"]["repository"]) / "Origindata"
        if (self.project / "Origindata").resolve() != raw.resolve():
            raise ValueError("Snapshot original-data link has an unexpected target")
        relative = ["positions/positions_vla.txt", "events/range/RangeEventS5/SproulToVLA.S5.txt", "environments/ctds/i9605.prn"]
        channels = sorted((raw / "events/S5/vla_matfiles").glob("S5_VLA_NO_*.mat"))
        if len(channels) != 21:
            raise ValueError(f"Expected 21 S5 VLA MAT files, found {len(channels)}")
        relative += [p.relative_to(raw).as_posix() for p in channels]
        inputs = [{"path": name, "size": (raw/name).stat().st_size,
                   "sha256": file_hash(raw/name)} for name in relative]
        identity = {"schema_version": 1, "pack": PACK, "inputs": inputs,
                    "dataset": self.execution_config["dataset"], "rbd": self.execution_config["rbd"],
                    "source_sha256": self.manifest["source_sha256"], "uv_lock_sha256": self.lock_hash}
        write_json(self.run / "data_identity.json", identity)
        return {"inputs": len(inputs), "identity_sha256": file_hash(self.run / "data_identity.json")}

    def demo(self) -> dict:
        output = self.run / "reports" / "demo"
        output.mkdir(parents=True, exist_ok=True)
        suite = "experiment" if self.request["suite"] == "selected" else self.request["suite"]
        v = self.execution_config["validation"]
        timeout = v["quick_timeout_s"] if suite == "quick" else v["generation_timeout_s"] + v["training_timeout_s"]
        result = self.json_command("demo", [self.python, "-B", "-m", "scripts.MoE_learn.localization_networks.demo",
                                  "--config", str(self.run / "resolved-config.json"), "--output", str(output),
                                  "--suite", suite], output / "demo-report.json", timeout)
        if result.get("pack") != PACK:
            raise ValueError("Demo report pack identity mismatch")
        if suite != "quick" and not (output / "checkpoint.pt").is_file():
            raise ValueError("Completed demo is missing its restorable checkpoint")
        self.report["demo"] = result
        return result

    def export(self, terminal_report: dict) -> None:
        target = self.run / "export" / "data"
        figures = self.run / "export" / "figure"
        target.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "request.json", "resolved-config.json", "environment.json", "data_identity.json"):
            if (self.run / name).is_file():
                shutil.copyfile(self.run / name, target / name)
        for directory in (self.run / "logs", self.run / "reports"):
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(self.run)
                suffix = path.suffix.lower()
                if suffix in {".pdf", ".png", ".svg"}:
                    dest = figures / relative
                elif suffix in {".json", ".log", ".csv"} or (suffix == ".pt" and path.parent == self.run / "reports" / "demo"):
                    dest = target / relative
                else:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
        write_json(target / "report.json", terminal_report)
        # Logs can keep changing until the worker exits; rsync checksums cover
        # their transfer, while this inventory covers immutable result artifacts.
        inventory = []
        for group, root in (("data", target), ("figure", figures)):
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if path.is_file() and relative.parts[0] != "logs" and path.name != "artifact-manifest.json":
                    inventory.append({"group": group, "path": relative.as_posix(),
                                      "size": path.stat().st_size, "sha256": file_hash(path)})
        write_json(target / "artifact-manifest.json", {"schema_version": 1, "pack": PACK,
                   "run_id": self.run.name, "files": inventory})

    def execute(self) -> int:
        self.save()
        lock = None
        code = 1
        planned_phases = {"snapshot_integrity"}
        try:
            self.phase("snapshot_integrity", self.setup)
            suite = self.request["suite"]
            planned_phases = expected_phases(suite, self.request.get("selection"))
            workspace = Path(self.config["remote"]["workspace"])
            lock = (workspace / "validation.lock").open("a")
            self.report["status"] = "WAITING_FOR_RESOURCE"
            self.save()
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.report["status"] = "RUNNING"
            self.save()
            self.phase("environment", self.environment)
            if suite != "quick":
                self.phase("original_data_identity", self.data_identity)
            self.phase("demo", self.demo)
            self.phase("source_unchanged_after_validation", lambda: verify_snapshot(self.project, self.manifest))
            self.report.update(status="PASS", exit_code=0)
            code = 0
        except BaseException as exc:
            self.report.update(status="FAIL", exit_code=1, error=f"{type(exc).__name__}: {exc}")
            print(self.report["error"], flush=True)
        finally:
            self.report["not_run"] = sorted(planned_phases - {row["name"] for row in self.report["checks"]})
            self.report["finished_at"] = stamp()
            terminal_report = {**self.report, "updated_at": stamp()}
            self.report["status"] = "EXPORTING"
            self.save()
            try:
                self.export(terminal_report)
                # Publish terminal status only after immutable exports exist.
                self.report = terminal_report
                write_json(self.run / "report.json", self.report)
            except BaseException as exc:
                self.report.update(status="FAIL", exit_code=1, error=f"Result export failed: {type(exc).__name__}: {exc}")
                self.save()
                write_json(self.run / "export/data/report.json", self.report)
                code = 1
            if lock is not None:
                lock.close()
        return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(f"Remote worker received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    return Worker(args.run_dir).execute()


if __name__ == "__main__":
    raise SystemExit(main())
