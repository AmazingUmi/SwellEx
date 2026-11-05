"""Stdlib-only model catalog for MoE_data_scale."""
from __future__ import annotations

METHODS = ("RBD", "ELM", "SCM", "RBD_SCM")
ARCHITECTURES = (
    "complex_MoE",
    "complex_MoE_e1",
    "complex_MoE_e2",
    "complex_MoE_e4",
    "complex_MoE_e8",
)
_PACKAGES = {"RBD": "rbd", "ELM": "elm", "SCM": "scm", "RBD_SCM": "rbd_scm"}
_MODELS = {
    "complex_MoE": "complex_moe_range",
    **{f"complex_MoE_e{count}": f"complex_moe_e{count}_range" for count in (1, 2, 4, 8)},
}


def package_name(method: str) -> str:
    """Return the canonical Python package for a public feature method."""
    if not isinstance(method, str) or method not in _PACKAGES:
        raise ValueError(f"METHOD must be one of {', '.join(METHODS)}")
    return _PACKAGES[method]


def entry_script(selection: dict) -> str:
    """Resolve an already canonical model selection to its sole CLI entry."""
    selected = validate_selection(selection)
    package = package_name(selected["method"])
    return f"scripts/network_training/MoE_data_scale/{package}/main.py"


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
