"""Stdlib-only model catalog: feature methods x network families -> model names and entry scripts."""
from __future__ import annotations

from scripts.data_processing.input_contract import METHODS
from scripts.network_training.grnn.bindings import model_name, validate_method

ARCHITECTURES = ("complex_CNN", "ResNet18", "ResNet50", "GRNN")
NEURAL_MODELS = {
    "complex_CNN": "complex_cnn_range",
    "ResNet18": "resnet18_range",
    "ResNet50": "resnet50_range",
}
# Neural registry names: RBD is unprefixed, SCM carries `scm_` (checkpoint/artifact contract).
MODEL_PREFIX = {"RBD": "", "SCM": "scm_"}
NEURAL_ENTRY = "scripts/network_training/neural/main.py"
GRNN_ENTRY = "scripts/network_training/grnn/main.py"


def neural_model_name(method: str, architecture: str) -> str:
    return MODEL_PREFIX[validate_method(method)] + NEURAL_MODELS[architecture]


def entry_script(selection: dict) -> str:
    """Resolve an already canonical model selection to its sole CLI entry."""
    selected = validate_selection(selection)
    return GRNN_ENTRY if selected["architecture"] == "GRNN" else NEURAL_ENTRY


def resolve_selection(method: str, architecture: str) -> dict:
    validate_method(method)
    if not isinstance(architecture, str) or architecture not in ARCHITECTURES:
        raise ValueError(f"ARCH must be one of {', '.join(ARCHITECTURES)}")
    model = model_name(method) if architecture == "GRNN" else neural_model_name(method, architecture)
    return {"method": method, "architecture": architecture, "model": model}


def validate_selection(selection: object) -> dict:
    if not isinstance(selection, dict) or set(selection) != {"method", "architecture", "model"}:
        raise ValueError("Selection must contain exactly method, architecture, model")
    canonical = resolve_selection(selection["method"], selection["architecture"])
    if selection != canonical:
        raise ValueError("Selection model does not match its method and architecture")
    return canonical
