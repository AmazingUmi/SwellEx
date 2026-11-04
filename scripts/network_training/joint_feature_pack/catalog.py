"""Stdlib-only model catalog for joint_feature_pack."""
from __future__ import annotations

from scripts.data_processing.joint_feature_pack.contract import FUSION_STRATEGIES as FUSIONS, METHODS
from scripts.network_training.joint_feature_pack.grnn.bindings import model_name, validate_method

ARCHITECTURES = ("complex_CNN", "real_CNN", "ResNet18", "ResNet50", "GRNN")
_MODELS = {
    "complex_CNN": "joint_complex_cnn_range",
    "real_CNN": "joint_real_cnn_range",
    "ResNet18": "joint_resnet18_range",
    "ResNet50": "joint_resnet50_range",
}


def package_name(method: str) -> str:
    validate_method(method)
    return "joint"


def resolve_selection(method: str, architecture: str) -> dict:
    validate_method(method)
    if not isinstance(architecture, str) or architecture not in ARCHITECTURES:
        raise ValueError(f"ARCH must be one of {', '.join(ARCHITECTURES)}")
    return {
        "method": method,
        "architecture": architecture,
        "model": model_name(method) if architecture == "GRNN" else _MODELS[architecture],
    }


def validate_selection(selection: object) -> dict:
    if not isinstance(selection, dict) or set(selection) != {"method", "architecture", "model"}:
        raise ValueError("Selection must contain exactly method, architecture, model")
    canonical = resolve_selection(selection["method"], selection["architecture"])
    if selection != canonical:
        raise ValueError("Selection model does not match its method and architecture")
    return canonical


def entry_script(selection: dict) -> str:
    selected = validate_selection(selection)
    package = "grnn" if selected["architecture"] == "GRNN" else "joint"
    return f"scripts/network_training/joint_feature_pack/{package}/main.py"
