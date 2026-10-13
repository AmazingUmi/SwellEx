"""Focused joint-pack checks, executed in the frozen remote environment."""
from __future__ import annotations
import ast
import os
from pathlib import Path
import sys

from scripts.joint_feature_pack.localization_networks.common.paths import PROJECT_ROOT as ROOT
from scripts.joint_feature_pack.localization_networks.workflow.processes import run_process
from scripts.joint_feature_pack.localization_networks.workflow.tasks import METHODS, ARCHITECTURES, resolve_selection, model_plan
from scripts.joint_feature_pack.localization_networks.validation.reporting import require

PACK_ROOT = ROOT / "scripts/joint_feature_pack"


def syntax_check():
    paths = list(PACK_ROOT.rglob("*.py"))
    for path in paths:
        if not path.name.startswith("._"):
            ast.parse(path.read_text(), filename=str(path))
    return {"files": len(paths)}


def package_tests_check(relative):
    directory = PACK_ROOT / relative
    require(directory.is_dir() and list(directory.glob("test*.py")), f"Missing tests: {directory}")
    result = run_process([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(directory), "-p", "test*.py", "-v"],
                         cwd=ROOT, text=True, capture_output=True, timeout=300)
    require(result.returncode == 0, f"{relative}: exit={result.returncode}\n{result.stdout}\n{result.stderr}")
    return {"stdout": result.stdout, "stderr": result.stderr}


def runtime_check(device):
    import torch
    torch.set_num_threads(max(1, min(4, int(os.environ.get("OMP_NUM_THREADS", "4")))))
    chosen = torch.device(device)
    if chosen.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable; no CPU fallback")
    actual = torch.ones(1, device=chosen).device
    require(actual.type == chosen.type, "Requested device differs from actual")
    return {"device": str(actual), "torch": torch.__version__, "cpu_threads": torch.get_num_threads()}


def registry_check():
    from scripts.joint_feature_pack.localization_networks.joint.model import MODEL_REGISTRY
    from scripts.packs import get_pack
    require(tuple(METHODS) == get_pack("joint_feature_pack").methods, "Dispatcher method list differs")
    for method in METHODS:
        require({resolve_selection(method, a)["model"] for a in ARCHITECTURES if a != "GRNN"} == set(MODEL_REGISTRY),
                "Execution plan differs from joint model registry")
    require(len(model_plan("experiment")) == 35, "Expected 35 independent input/model cases")
    return {"methods": METHODS, "models": list(MODEL_REGISTRY), "experiment_jobs": 35}


def model_check(method, architecture, device):
    import torch
    from scripts.joint_feature_pack.localization_networks.joint import model
    heights = {"RBD": 21, "ELM": 210, "SCM": 231}
    height = sum(heights[m] for m in method.split("_"))
    name = resolve_selection(method, architecture)["model"]
    config = model.build_model_config(name, (2, height, 22))
    net = model.build_model(name, config).to(device)
    x = torch.randn(2, 2, height, 22, device=device)
    y = net(x)
    require(tuple(y.shape) == (2, 1) and bool(torch.isfinite(y).all()), "Invalid forward output")
    y.square().mean().backward()
    gradients = [p.grad for p in net.parameters() if p.requires_grad]
    require(any(g is not None for g in gradients) and all(g is None or bool(torch.isfinite(g).all()) for g in gradients), "Invalid model gradients")
    return {"method": method, "architecture": architecture, "shape": list(x.shape),
            "parameters": sum(p.numel() for p in net.parameters())}


def quick(report, device):
    report.check("python.syntax", syntax_check)
    usable = report.check("runtime.device", lambda: runtime_check(device))
    report.check("workflow.registry", registry_check)
    for relative in ("localization_networks/workflow/tests", "localization_networks/joint/tests",
                     "localization_networks/common/tests", "localization_networks/grnn/tests",
                     "data_processing/tests", "visualization/tests"):
        report.check(relative.replace("/", "."), lambda p=relative: package_tests_check(p))
    for method in METHODS:
        for architecture in ARCHITECTURES[:-1]:
            def check(m=method, a=architecture):
                require(usable, "Requested device failed")
                return model_check(m, a, device)
            report.check(f"model.{method}.{architecture}", check)
