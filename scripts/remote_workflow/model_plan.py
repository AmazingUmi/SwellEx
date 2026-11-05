"""Canonical execution plans (suite -> ordered model list) shared by the stdlib dispatcher and worker."""
from __future__ import annotations

from scripts.network_training.catalog import (
    ARCHITECTURES, METHODS, NEURAL_MODELS, entry_script, resolve_selection, validate_selection,
)
from scripts.remote_workflow.run_identity import validate_filters

SUITES = ("quick", "smoke", "experiment", "selected")
__all__ = ["ARCHITECTURES", "METHODS", "SUITES", "entry_script", "expected_phases", "model_plan",
           "resolve_selection", "validate_selection"]


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
    architectures = ("complex_CNN",) if suite == "smoke" else tuple(NEURAL_MODELS)
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
