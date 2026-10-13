"""Canonical execution plans shared by the stdlib dispatcher and worker."""
from __future__ import annotations

from scripts.base_pack.localization_networks.grnn.bindings import METHODS, model_name, validate_method

from scripts.packs import NETWORKS
from scripts.run_identity import validate_filters

SUITES = ("quick", "smoke", "experiment", "selected")
ARCHITECTURES = NETWORKS
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
    return f"scripts/base_pack/localization_networks/{package}/main.py"


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


def model_plan(suite: str, selection: dict | None = None) -> list[dict]:
    if not isinstance(suite, str) or suite not in SUITES:
        raise ValueError(f"Unknown execution suite: {suite}")
    if suite == "selected":
        filters = validate_filters("base_pack", selection)
        methods = (filters["method"],) if filters["method"] else METHODS
        networks = (filters["network"],) if filters["network"] else ARCHITECTURES
        return [resolve_selection(method, network) for method in methods for network in networks]
    if selection is not None:
        raise ValueError("Selection is allowed only for the selected suite")
    if suite == "quick":
        return []
    architectures = ("complex_CNN",) if suite == "smoke" else tuple(_MODELS)
    return [resolve_selection(method, architecture) for method in METHODS for architecture in architectures] + [resolve_selection(method, "GRNN") for method in METHODS]


def expected_phases(suite: str, selection: dict | None = None) -> set[str]:
    plan = model_plan(suite, selection)
    phases = {"snapshot_integrity", "environment", "source_unchanged_after_validation"}
    if suite != "selected":
        phases.update({"python_quick", "data_numerical"})
    if suite != "quick":
        phases.update({"original_data_identity", "data_generation", "dataset_contract", "visualization"})
        phases.update(f"evaluate_{item['method']}_{item['model']}" for item in plan)
    if suite in {"smoke", "experiment"}:
        phases.add("checkpoint_resume")
    return phases
