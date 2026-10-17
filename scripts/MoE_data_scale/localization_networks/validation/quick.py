"""CUDA synthetic architecture validation; never trains on SwellEx data."""
from __future__ import annotations
import importlib
from pathlib import Path
import subprocess
import sys
import unittest
from .reporting import require

PACKAGE = "scripts.MoE_data_scale.localization_networks"
METHODS = ("RBD", "ELM", "SCM", "RBD_SCM")
HEIGHTS = {"RBD": 21, "ELM": 210, "SCM": 231, "RBD_SCM": 252}


def workflow_tests_check():
    root = Path(__file__).resolve().parents[4]
    command = [sys.executable, "-B", "-m", "unittest",
               "scripts.MoE_data_scale.localization_networks.workflow.tests.test_moe_workflow",
               "scripts.MoE_data_scale.localization_networks.workflow.tests.test_study",
               "scripts.MoE_data_scale.localization_networks.workflow.tests.test_study_worker",
               "scripts.MoE_data_scale.localization_networks.rbd_scm.tests.test_contract",
               "scripts.tests.test_moe_data_scale_pack",
               "scripts.MoE_data_scale.visualization.tests.test_architecture_metadata",
               "scripts.MoE_data_scale.visualization.tests.test_study_results", "-v"]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=180)
    require(result.returncode == 0, result.stdout + "\n" + result.stderr)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
            "scope": "pack_dispatch_and_snapshot_contracts"}


def model_module(method):
    return importlib.import_module(f"{PACKAGE}.{method.lower()}.network.model")


def device_check(device):
    import torch
    chosen = torch.device(device)
    require(chosen.type == "cuda", "Architecture validation requires explicit CUDA; CPU fallback is forbidden")
    require(torch.cuda.is_available(), "CUDA requested but unavailable; no CPU fallback")
    tensor = torch.ones(1, device=chosen)
    require(tensor.is_cuda, f"device mismatch: {tensor.device}")
    return {"requested": device, "actual": str(tensor.device), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(chosen)}


def registry_check():
    from scripts.MoE_data_scale.localization_networks.workflow.tasks import ARCHITECTURES, resolve_selection
    expected = {"complex_MoE", "complex_MoE_e1", "complex_MoE_e2", "complex_MoE_e4", "complex_MoE_e8"}
    require(set(ARCHITECTURES) == expected, "Public MoE architecture inventory differs")
    rows = []
    for method in METHODS:
        module = model_module(method)
        selections = [resolve_selection(method, name) for name in ARCHITECTURES]
        require({item["model"] for item in selections} == set(module.MODEL_REGISTRY), "Registry/public selection mismatch")
        require(len(module.MODEL_REGISTRY) == 5, "Each method must register five MoE variants")
        rows.extend(selections)
    return {"count": len(rows), "selections": rows}


def model_check(method, name, device, *, base_channels=16, frequencies=22, batch_size=2):
    import torch
    from scripts.MoE_data_scale.localization_networks.common.scaling import DEFAULT_SCALING_POLICY
    torch.manual_seed(2026)
    module = model_module(method)
    shape = (2, HEIGHTS[method], frequencies)
    config = module.build_model_config(name, shape, base_channels=base_channels, dropout=0.0,
                                       train_count=600, scaling_policy=dict(DEFAULT_SCALING_POLICY))
    model = module.build_model(name, config).to(device)
    model.train()
    x = torch.randn(batch_size, *shape, device=device, requires_grad=True)
    prediction, aux = model.forward_with_aux(x)
    require(prediction.is_cuda and tuple(prediction.shape) == (batch_size, 1), "CUDA [batch,1] output required")
    require(bool(torch.isfinite(prediction).all()), "Prediction is nonfinite")
    targets = torch.linspace(-0.6, 0.4, batch_size, device=device).unsqueeze(1)
    loss = torch.nn.functional.mse_loss(prediction, targets)
    objective = loss + 0.01 * aux["balance_loss"]
    require(aux["balance_loss"].ndim == 0 and aux["balance_loss"].requires_grad, "Auxiliary loss must be differentiable scalar")
    objective.backward()
    require(x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.abs().sum() > 0), "Input gradients missing/nonfinite/zero")
    selected = set(aux["selected_experts"].flatten().tolist())
    for index, expert in enumerate(model.experts):
        gradients = [p.grad for p in expert.parameters()]
        if index in selected:
            require(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients), f"Selected expert {index} missing finite gradient")
            require(any(bool(g.abs().sum() > 0) for g in gradients), "Selected expert gradients all zero")
        else:
            require(all(g is None for g in gradients), "Unselected expert received a gradient")
    require(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.stem.parameters()), "Stem gradients missing")
    if model.router is not None:
        require(model.router.weight.grad is not None and bool(model.router.weight.grad.abs().sum() > 0), "Router gradients missing/zero")
    count = sum(p.numel() for p in model.parameters())
    require(count == config.parameter_count, "Frozen parameter count differs from actual parameters")
    model.eval()
    with torch.no_grad():
        ordinary = model(x.detach())
        with_aux, _ = model.forward_with_aux(x.detach())
    require(torch.equal(ordinary, with_aux), "forward and forward_with_aux differ in evaluation")
    return {"input_kind": "synthetic_random", "method": method, "model": name,
            "input_shape": list(x.shape), "output_shape": list(prediction.shape),
            "base_channels": base_channels, "num_experts": config.num_experts,
            "selected_experts": sorted(selected), "parameters": count,
            "active_parameters_per_sample": config.active_parameter_count,
            "regression_loss": float(loss.detach()), "balance_loss": float(aux["balance_loss"].detach()),
            "device": str(prediction.device), "scientific_performance_evidence": False}


def quick(report, device):
    """Run every numerical check on the exact requested CUDA device."""
    report.data.update({"pack": "MoE_data_scale", "scope": "synthetic_architecture_only",
                        "input_kind": "synthetic_random", "swellex_training": False,
                        "scientific_performance_evidence": False, "requested_device": device})
    if not report.check("runtime.cuda", lambda: device_check(device)):
        return
    import torch
    torch.set_num_threads(4)
    report.check("workflow.pack_contracts", workflow_tests_check)
    from .datasets import validator_controls_check
    report.check("datasets.raw_generation_contracts", validator_controls_check)
    report.check("registry.twenty_variants", registry_check)
    for method in METHODS:
        for index, name in enumerate(model_module(method).MODEL_REGISTRY):
            report.check(f"architecture.{method}.{name}",
                         lambda m=method, n=name, i=index: model_check(m, n, device, frequencies=22 + i % 2))
    from scripts.MoE_data_scale.localization_networks.workflow.tasks import resolve_selection
    name = resolve_selection("SCM", "complex_MoE_e4")["model"]
    report.check("architecture.SCM.default_channels_batch64_full_shape_e4",
                 lambda: model_check("SCM", name, device, base_channels=16, frequencies=23, batch_size=64))
    joint_name = resolve_selection("RBD_SCM", "complex_MoE_e4")["model"]
    report.check("architecture.RBD_SCM.default_channels_batch64_full_shape_e4",
                 lambda: model_check("RBD_SCM", joint_name, device, base_channels=16, frequencies=22, batch_size=64))
    suite = unittest.TestSuite()
    for name in ("common.tests.test_moe_contracts", "rbd_scm.tests.test_joint_input"):
        tests = importlib.import_module(f"{PACKAGE}.{name}")
        tests.REQUESTED_DEVICE = device
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(tests))
    def cases(items):
        for item in items:
            if isinstance(item, unittest.TestSuite):
                yield from cases(item)
            else:
                yield item
    for case in cases(suite):
        def run(test=case):
            result = unittest.TestResult()
            test.run(result)
            require(result.testsRun == 1 and not result.skipped, "Contract test did not execute")
            require(result.wasSuccessful(), "\n".join(message for _, message in result.errors + result.failures))
            return {"tests_run": 1, "device": device, "input_kind": "synthetic_random"}
        report.check("contracts." + case._testMethodName, run)
