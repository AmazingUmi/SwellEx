"""Execution plans for seven input combinations and unchanged regressors."""
from __future__ import annotations

from scripts.joint_feature_pack.localization_networks.grnn.bindings import METHODS, model_name, validate_method

from scripts.packs import NETWORKS
from scripts.run_identity import validate_filters

SUITES = ("quick", "smoke", "experiment", "selected")
ARCHITECTURES = NETWORKS
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
    return {"method": method, "architecture": architecture,
            "model": model_name(method) if architecture == "GRNN" else _MODELS[architecture]}


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
    return f"scripts/joint_feature_pack/localization_networks/{package}/main.py"


def model_plan(suite: str, selection: dict | None = None) -> list[dict]:
    if not isinstance(suite, str) or suite not in SUITES:
        raise ValueError(f"Unknown execution suite: {suite}")
    if suite == "selected":
        filters = validate_filters("joint_feature_pack", selection)
        methods = (filters["method"],) if filters["method"] else METHODS
        networks = (filters["network"],) if filters["network"] else ARCHITECTURES
        return [resolve_selection(method, network) for method in methods for network in networks]
    if selection is not None:
        raise ValueError("Selection is allowed only for the selected suite")
    if suite == "quick":
        return []
    architectures = ("complex_CNN",) if suite == "smoke" else tuple(_MODELS)
    return [resolve_selection(m, a) for m in METHODS for a in architectures] + [resolve_selection(m, "GRNN") for m in METHODS]


def expected_phases(suite: str, selection: dict | None = None) -> set[str]:
    plan = model_plan(suite, selection)
    phases = {"snapshot_integrity", "environment", "source_unchanged_after_validation"}
    if suite != "selected":
        phases.update({"python_quick", "data_numerical"})
    if suite != "quick":
        phases.update({"original_data_identity", "data_generation", "dataset_contract", "joint_datasets", "visualization"})
        phases.update(f"evaluate_{item['method']}_{item['model']}" for item in plan)
    if suite in {"smoke", "experiment"}:
        phases.add("checkpoint_resume")
    return phases
