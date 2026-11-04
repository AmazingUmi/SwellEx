"""Stdlib-only model catalog for base_pack."""
from __future__ import annotations

from scripts.network_training.base_pack.grnn.bindings import METHODS, model_name, validate_method

ARCHITECTURES = ("complex_CNN", "real_CNN", "ResNet18", "ResNet50", "GRNN")
_PACKAGES = {"RBD": "rbd", "ELM": "elm", "SCM": "scm"}
_MODELS = {
    "complex_CNN": "complex_cnn_range",
    "real_CNN": "real_cnn_range",
    "ResNet18": "resnet18_range",
    "ResNet50": "resnet50_range",
}


def package_name(method: str) -> str:
    """Return the canonical Python package for a public feature method."""
    if not isinstance(method, str) or method not in _PACKAGES:
        raise ValueError(f"METHOD must be one of {', '.join(METHODS)}")
    return _PACKAGES[method]


def entry_script(selection: dict) -> str:
    """Resolve an already canonical model selection to its sole CLI entry."""
    selected = validate_selection(selection)
    package = "grnn" if selected["architecture"] == "GRNN" else package_name(selected["method"])
    return f"scripts/network_training/base_pack/{package}/main.py"


def resolve_selection(method: str, architecture: str) -> dict:
    validate_method(method)
    if not isinstance(architecture, str) or architecture not in ARCHITECTURES:
        raise ValueError(f"ARCH must be one of {', '.join(ARCHITECTURES)}")
    if architecture == "GRNN":
        model = model_name(method)
    else:
        prefix = "" if method == "RBD" else method.lower() + "_"
        model = prefix + _MODELS[architecture]
    return {"method": method, "architecture": architecture, "model": model}


def validate_selection(selection: object) -> dict:
    if not isinstance(selection, dict) or set(selection) != {"method", "architecture", "model"}:
        raise ValueError("Selection must contain exactly method, architecture, model")
    canonical = resolve_selection(selection["method"], selection["architecture"])
    if selection != canonical:
        raise ValueError("Selection model does not match its method and architecture")
    return canonical
