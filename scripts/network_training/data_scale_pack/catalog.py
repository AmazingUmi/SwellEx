"""Stdlib-only model catalog for data_scale_pack."""
from __future__ import annotations

METHODS = ("RBD", "ELM", "SCM")
ARCHITECTURES = (
    "complex_CNN",
    "complex_CNN_small",
    "complex_CNN_medium",
    "complex_CNN_large",
    "complex_CNN_xlarge",
    "complex_CNN_baseline",
)
_PACKAGES = {"RBD": "rbd", "ELM": "elm", "SCM": "scm"}
_MODELS = {
    "complex_CNN": "complex_cnn_range",
    "complex_CNN_small": "complex_cnn_small_range",
    "complex_CNN_medium": "complex_cnn_medium_range",
    "complex_CNN_large": "complex_cnn_large_range",
    "complex_CNN_xlarge": "complex_cnn_xlarge_range",
    "complex_CNN_baseline": "complex_cnn_baseline_range",
}
BUDGET_ARCHITECTURES = tuple(name for name in _MODELS if name != "complex_CNN")


def package_name(method: str) -> str:
    """Return the canonical Python package for a public feature method."""
    if not isinstance(method, str) or method not in _PACKAGES:
        raise ValueError(f"METHOD must be one of {', '.join(METHODS)}")
    return _PACKAGES[method]


def entry_script(selection: dict) -> str:
    """Resolve an already canonical model selection to its sole CLI entry."""
    selected = validate_selection(selection)
    package = package_name(selected["method"])
    return f"scripts/network_training/data_scale_pack/{package}/main.py"


def resolve_selection(method: str, architecture: str) -> dict:
    package_name(method)
    if not isinstance(architecture, str) or architecture not in ARCHITECTURES:
        raise ValueError(f"ARCH must be one of {', '.join(ARCHITECTURES)}")
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
