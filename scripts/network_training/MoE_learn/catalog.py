"""Stdlib-only model catalog for MoE_learn."""
from __future__ import annotations

METHODS = ("RBD_SCM",)
ARCHITECTURES = ("complex_MoE_learn",)
DEMO_MODULE = "scripts.network_training.MoE_learn.demo"


def resolve_selection(method: str, architecture: str) -> dict:
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"METHOD must be one of {', '.join(METHODS)}")
    if not isinstance(architecture, str) or architecture not in ARCHITECTURES:
        raise ValueError(f"ARCH must be one of {', '.join(ARCHITECTURES)}")
    return {"method": method, "architecture": architecture, "model": "rbd_scm_complex_moe_learn_range"}


def validate_selection(selection: object) -> dict:
    if not isinstance(selection, dict) or set(selection) != {"method", "architecture", "model"}:
        raise ValueError("Selection must contain exactly method, architecture, model")
    canonical = resolve_selection(selection["method"], selection["architecture"])
    if selection != canonical:
        raise ValueError("Selection model does not match its method and architecture")
    return canonical


def entry_script(selection: dict | None = None) -> str:
    if selection is not None:
        validate_selection(selection)
    return "scripts/network_training/MoE_learn/rbd_scm/main.py"
